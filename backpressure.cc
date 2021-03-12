#include "legion.h"
#include "mappers/logging_wrapper.h"
#include "mappers/null_mapper.h"
#include <chrono>
#include <thread>

Realm::Logger log_app("app");

using namespace Legion;

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_TEST,
};

enum FieldIDs {
  FID_DATA = 10,
};

class BackpressureMapper : public Mapping::NullMapper {
 public:
  BackpressureMapper(Mapping::MapperRuntime *_rt, Machine _machine)
    : NullMapper(_rt, _machine)
  {
  }

  const char *get_mapper_name(void) const
  {
    return "BackpressureMapper";
  }

  MapperSyncModel get_mapper_sync_model(void) const
  {
    return SERIALIZED_REENTRANT_MAPPER_MODEL;
  }

  bool request_valid_instances(void) const
  {
    return false;
  }

  void select_steal_targets(const Mapping::MapperContext ctx,
			    const SelectStealingInput& input,
			    SelectStealingOutput& output)
  {
    // no stealing
  }

  void select_task_options(const Mapping::MapperContext ctx,
			   const Task& task, TaskOptions& output)
  {
    // defaults
  }

  void select_tasks_to_map(const Mapping::MapperContext ctx,
			   const SelectMappingInput& input,
			   SelectMappingOutput& output)
  {
    output.map_tasks.insert(input.ready_tasks.begin(),
			    input.ready_tasks.end());
  }

  void map_task(const Mapping::MapperContext ctx,
		const Task& task,
		const MapTaskInput& input, MapTaskOutput& output)
  {
    Machine::MemoryQuery mq(machine);
    mq.only_kind(Memory::SYSTEM_MEM);
    assert(mq.count() == 1);
    Memory mem = mq.first();

    Machine::ProcessorQuery pq(machine);
    pq.only_kind(Processor::LOC_PROC);
    assert(pq.count() == 1);
    Processor proc = pq.first();

    for (size_t i = 0; i < task.regions.size(); i++) {
      const RegionRequirement& req = task.regions[i];
      LayoutConstraintSet constraints;
      constraints.add_constraint(FieldConstraint(req.privilege_fields, false /*contiguous*/));
      std::vector<LogicalRegion> regions(1, req.region);
      Mapping::PhysicalInstance inst;
      bool created;
      while (!runtime->find_or_create_physical_instance(ctx, mem, constraints, regions, inst, created)) {
        log_app.info() << "Failed allocation for requirement " << i << " of task " << task.get_unique_id() << ", preparing to sleep"; std::cout.flush();
        // Relase any instances we've acquired along the way
        runtime->release_instances(ctx, output.chosen_instances);
        // Get the event to wait on (a single event for all calls waiting on the same memory)
        Mapping::MapperEvent event;
        runtime->disable_reentrant(ctx);
        if (!sysmem_wait_event_.exists() || runtime->has_mapper_event_triggered(ctx, sysmem_wait_event_)) {
          sysmem_wait_event_ = runtime->create_mapper_event(ctx);
        }
        event = sysmem_wait_event_;
        runtime->enable_reentrant(ctx);
        // Sleep on the event
        log_app.info() << "Failed allocation for requirement " << i << " of task " << task.get_unique_id() << ", going to sleep"; std::cout.flush();
        runtime->wait_on_mapper_event(ctx, event);
        log_app.info() << "Woke up, retrying allocation for requirement " << i << " of task " << task.get_unique_id(); std::cout.flush();
        // Re-acquire all previously acquired instances
        bool reacquired = runtime->acquire_instances(ctx, output.chosen_instances);
        assert(reacquired);
        // BUG: If we go through this loop the mapping succeeds, but the task never starts.
      }
      log_app.info() << "Allocation success for requirement " << i << " of task " << task.get_unique_id(); std::cout.flush();
      output.chosen_instances[i].push_back(inst);
    }

    output.target_procs.push_back(proc);

    std::vector<VariantID> valid_variants;
    runtime->find_valid_variants(ctx, task.task_id, valid_variants, proc.kind());
    assert(valid_variants.size() == 1);
    output.chosen_variant = valid_variants[0];
    log_app.info() << "Done mapping task " << task.get_unique_id(); std::cout.flush();
  }

  // HACK: A callback that will be invoked after a task has run and its memory
  // released. Ideally the runtime would wake up the mapper once some storage
  // becomes available on a target memory.
  void select_tunable_value(const Mapping::MapperContext ctx,
                            const Task &task,
                            const SelectTunableInput &input,
                            SelectTunableOutput &output)
  {
    runtime->disable_reentrant(ctx);
    if (sysmem_wait_event_.exists() && !runtime->has_mapper_event_triggered(ctx, sysmem_wait_event_)) {
      runtime->trigger_mapper_event(ctx, sysmem_wait_event_);
    }
    runtime->enable_reentrant(ctx);
    runtime->pack_tunable(1729, output);
    log_app.info() << "Triggered mapper event"; std::cout.flush();
  }

  void configure_context(const Mapping::MapperContext ctx,
                         const Task &task,
                         ContextConfigOutput &output)
  {
    // defaults
  }

 private:
  Mapping::MapperEvent sysmem_wait_event_;
};

void mapper_registration(Machine machine, Runtime *rt,
                         const std::set<Processor> &local_procs)
{
  rt->replace_default_mapper(new Mapping::LoggingWrapper(new BackpressureMapper(rt->get_mapper_runtime(), machine)));
}

void task_test(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  log_app.info() << "task starting"; std::cout.flush();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  log_app.info() << "task exiting"; std::cout.flush();
}

void task_top_level(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int N = 3;
  std::vector<Future> results;
  // std::vector<LogicalRegion> _regions;
  for (int i = 0; i < N; ++i) {
    Rect<1> bounds(0, (1<<20)-1); // 1MB
    IndexSpace is = runtime->create_index_space(ctx, bounds);
    FieldSpace fs = runtime->create_field_space(ctx);
    FieldAllocator fsa = runtime->create_field_allocator(ctx, fs);
    fsa.allocate_field(1, FID_DATA);
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher launcher(TID_TEST, TaskArgument());
    launcher.add_region_requirement(
      RegionRequirement(lr, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr)
      .add_field(FID_DATA));
    results.push_back(runtime->execute_task(ctx, launcher));
    runtime->destroy_logical_region(ctx, lr);
    // Could instead wait for the task to return and delete the region then.
    // _regions.push_back(lr);
  }
  // HACK: After a task finishes (and its instance has hopefully been marked as
  // collectable), wake up the mapper to try the mapping again. This should
  // really wait on any task to finish, it won't necessarily be the next one in
  // launch order.
  for (int i = 0; i < N-1; ++i) {
    log_app.info() << "Waiting for next task to finish"; std::cout.flush();
    results[i].get_void_result();
    // BUG: If I destroy the task's region here then the instance is not collected.
    // runtime->destroy_logical_region(ctx, _regions[i]);
    log_app.info() << "Waiting a second for instance to be collected"; std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    log_app.info() << "Waking up the mapper"; std::cout.flush();
    runtime->select_tunable_value(ctx, 42);
  }
}

int main(int argc, char **argv)
{
  { TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<task_top_level>(registrar, "top_level");
    Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  }
  { TaskVariantRegistrar registrar(TID_TEST, "test");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<task_test>(registrar, "test");
  }
  Runtime::add_registration_callback(mapper_registration);
  return Runtime::start(argc, argv);
}
