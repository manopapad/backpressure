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

    for(size_t i = 0; i < task.regions.size(); i++) {
      const RegionRequirement& req = task.regions[i];
      LayoutConstraintSet constraints;
      constraints.add_constraint(FieldConstraint(req.privilege_fields, false /*contiguous*/));
      std::vector<LogicalRegion> regions(1, req.region);
      Mapping::PhysicalInstance inst;
      bool created;
      bool ok = runtime->find_or_create_physical_instance(ctx, mem, constraints, regions, inst, created);
      if (!ok) {
        log_app.error("failed allocation");
        assert(false);
      }
      output.chosen_instances[i].push_back(inst);
    }

    output.target_procs.push_back(proc);

    std::vector<VariantID> valid_variants;
    runtime->find_valid_variants(ctx, task.task_id, valid_variants, proc.kind());
    assert(valid_variants.size() == 1);
    output.chosen_variant = valid_variants[0];

    // output.postmap_task = true;
  }

  void configure_context(const Mapping::MapperContext ctx,
			 const Task& task, ContextConfigOutput& output)
  {
    // defaults
  }

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
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

void task_top_level(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int N = 3;
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
    runtime->execute_task(ctx, launcher);
    runtime->destroy_logical_region(ctx, lr);
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
