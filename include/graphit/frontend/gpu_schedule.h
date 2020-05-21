//
// Created by Ajay Brahmakshatriya 
//

#ifndef GRAPHIT_GPU_SCHEDULE
#define GRAPHIT_GPU_SCHEDULE

#include <assert.h>
#include <graphit/frontend/abstract_schedule.h>


namespace graphit {
namespace fir {
namespace gpu_schedule {


enum gpu_schedule_options {
	PUSH, 
	PULL, 
	FUSED, 
	UNFUSED,
	UNFUSED_BITMAP,
	UNFUSED_BOOLMAP,
	ENABLED,
	DISABLED,
	TWC,
	TWCE,
	WM,
	CM,
	STRICT,
	EDGE_ONLY,
	VERTEX_BASED,
	INPUT_VERTEXSET_SIZE,
	BITMAP,
	BOOLMAP,
	BLOCKED,
	UNBLOCKED,
};

class GPUSchedule {
	// Abstract class has no functions for now
public:
	// Virtual destructor to make the class polymorphic
	virtual ~GPUSchedule() = default;
};

class SimpleGPUSchedule: public GPUSchedule {

public:
	enum class pull_frontier_rep_type {
		BITMAP, 
		BOOLMAP
	};
	enum class direction_type {
		DIR_PUSH, 
		DIR_PULL
	};

	enum class frontier_creation_type {
		FRONTIER_FUSED, 
		UNFUSED_BITMAP,
		UNFUSED_BOOLMAP
	};

	enum class deduplication_type {
		DEDUP_DISABLED,
		DEDUP_ENABLED
	};
	enum class deduplication_strategy_type {
		DEDUP_FUSED,
		DEDUP_UNFUSED
	};

	enum class load_balancing_type {
		VERTEX_BASED,	
		TWC, 
		TWCE, 
		WM, 
		CM, 
		STRICT,
		EDGE_ONLY
	};
	
	enum class edge_blocking_type {
		BLOCKED,
		UNBLOCKED
	};

	enum class kernel_fusion_type {
		FUSION_DISABLED,
		FUSION_ENABLED
	};

	enum class boolean_type_type {
		BOOLMAP,
		BITMAP
	};

private:
public:
	direction_type direction;
	pull_frontier_rep_type pull_frontier_rep;
	frontier_creation_type frontier_creation;
	deduplication_type deduplication;
	deduplication_strategy_type deduplication_strategy;
	load_balancing_type load_balancing;
	edge_blocking_type edge_blocking;
	uint32_t edge_blocking_size;
	kernel_fusion_type kernel_fusion;
	boolean_type_type boolean_type;

	int32_t delta;
	
	SimpleGPUSchedule () {
		direction = direction_type::DIR_PUSH;
		pull_frontier_rep = pull_frontier_rep_type::BOOLMAP;
		frontier_creation = frontier_creation_type::FRONTIER_FUSED;
		deduplication = deduplication_type::DEDUP_DISABLED;
		load_balancing = load_balancing_type::VERTEX_BASED;
		edge_blocking = edge_blocking_type::UNBLOCKED;
		edge_blocking_size = 0;
		kernel_fusion = kernel_fusion_type::FUSION_DISABLED;
		delta = 1;
		boolean_type = boolean_type_type::BOOLMAP;
	}	

public:	
	void configDirection(enum gpu_schedule_options o, enum gpu_schedule_options r = BOOLMAP) {
		switch(o) {
			case PUSH:
				direction = direction_type::DIR_PUSH;
				break;
			case PULL:
				direction = direction_type::DIR_PULL;
				switch (r) {
					case BITMAP:
						pull_frontier_rep = pull_frontier_rep_type::BITMAP;
						break;
					case BOOLMAP:
						pull_frontier_rep = pull_frontier_rep_type::BOOLMAP;
						break;
					default:
						assert(false && "Invalid option for Pull Frontier representation\n");
						break;
				}
				break;
			default:
				assert(false && "Invalid option for configDirection");
				break;
		}	
	}
	
	void configFrontierCreation(enum gpu_schedule_options o) {
		switch(o) {
			case FUSED:
				frontier_creation = frontier_creation_type::FRONTIER_FUSED;
				break;
			case UNFUSED_BITMAP:
				frontier_creation = frontier_creation_type::UNFUSED_BITMAP;
				break;
			case UNFUSED_BOOLMAP:
				frontier_creation = frontier_creation_type::UNFUSED_BOOLMAP;
				break;
			default:
				assert(false && "Invalid option for configFrontierCreation");	
				break;
		}
	}

	void configDeduplication(enum gpu_schedule_options o, enum gpu_schedule_options l = UNFUSED) {
		switch(o) {
			case ENABLED:
				deduplication = deduplication_type::DEDUP_ENABLED;
				switch (l) {
					case FUSED:
						deduplication_strategy = deduplication_strategy_type::DEDUP_FUSED;
						break;
					case UNFUSED:
						deduplication_strategy = deduplication_strategy_type::DEDUP_UNFUSED;
						break;
					default:
						assert(false && "Invalid deduplication strategy\n");
						break;
				}
				break;
			case DISABLED:
				deduplication = deduplication_type::DEDUP_DISABLED;
				break;
			default:
				assert(false && "Invalid option for configDeduplication");
				break;
		}
	}

	void configLoadBalance(enum gpu_schedule_options o, enum gpu_schedule_options blocking = UNBLOCKED, int32_t blocking_size = 1) {
		switch(o) {
			case VERTEX_BASED:
				load_balancing = load_balancing_type::VERTEX_BASED;
				break;
			case TWC:
				load_balancing = load_balancing_type::TWC;
				break;
			case TWCE:
				load_balancing = load_balancing_type::TWCE;
				break;
			case WM:
				load_balancing = load_balancing_type::WM;
				break;
			case CM:
				load_balancing = load_balancing_type::CM;
				break;
			case STRICT:
				load_balancing = load_balancing_type::STRICT;
				break;
			case EDGE_ONLY:
				load_balancing = load_balancing_type::EDGE_ONLY;
				switch (blocking) {
					case BLOCKED:
						edge_blocking = edge_blocking_type::BLOCKED;
						edge_blocking_size = blocking_size;	
						break;
					case UNBLOCKED:
						edge_blocking = edge_blocking_type::UNBLOCKED;
						break;
					default:
						assert(false && "Invalid option for configLoadBalance");
						break;
				}
				break;
			default:
				assert(false && "Invalid option for configLoadBalance");
				break;
		}
	}
	
	void configKernelFusion(enum gpu_schedule_options o) {
		switch(o) {
			case ENABLED:
				kernel_fusion = kernel_fusion_type::FUSION_ENABLED;
				break;
			case DISABLED:
				kernel_fusion = kernel_fusion_type::FUSION_DISABLED;
				break;
			default:
				assert(false && "Invalid option for configKernelFusion");
				break;
		}
		
	}
	void configDelta(int32_t d) {
		if (d <= 0)
			assert(false && "Invalid option for configDelta");
		delta = d;
	}
	void configDelta(const char* d) {
		if (sscanf(d, "argv[%i]", &delta) != 1) {
			assert(false && "Invalid option for configDelta");
		}	
		delta *= -1;
	}
	void configBooleanType(enum gpu_schedule_options o) {
		switch(o) {
			case BOOLMAP:
				boolean_type = boolean_type_type::BOOLMAP;
				break;
			case BITMAP:
				boolean_type = boolean_type_type::BITMAP;
				break;
			default:
				assert(false && "Invalid option for configBooleanType");
				break;
		}
	}
	
};

class HybridGPUSchedule: public GPUSchedule {
private:
	// TODO: have separate alpha beta
public:	
	SimpleGPUSchedule s1;
	SimpleGPUSchedule s2;
	
	float threshold;
	int32_t argv_index;

	enum class hybrid_criteria {
		INPUT_VERTEXSET_SIZE
	};
	hybrid_criteria _hybrid_criteria;
private:	
public:	
	HybridGPUSchedule (enum gpu_schedule_options o, float t, SimpleGPUSchedule &_s1, SimpleGPUSchedule &_s2) {
		switch(o) {
			case INPUT_VERTEXSET_SIZE:
				_hybrid_criteria = hybrid_criteria::INPUT_VERTEXSET_SIZE;
				break;
			default:
				assert(false && "Invalid option for HybridGPUScheduleCriteria\n");
				break;
		}	
		threshold = t;
		s1 = _s1;
		s2 = _s2;
	}
	HybridGPUSchedule (enum gpu_schedule_options o, const char *t, SimpleGPUSchedule &_s1, SimpleGPUSchedule &_s2) {
		switch (o) {
			case INPUT_VERTEXSET_SIZE:
				_hybrid_criteria = hybrid_criteria::INPUT_VERTEXSET_SIZE;
				break;
			default:
				assert(false && "Invalid option for HybridGPUScheduleCriteria\n");
				break;
		}
		s1 = _s1;
		s2 = _s2;	
		if (sscanf(t, "argv[%i]", &argv_index) != 1) {
			assert(false && "Invalid threshold option\n");
		}
		threshold = -100;
	}
};


class SimpleGPUScheduleObject : public abstract_schedule::SimpleScheduleObject {
public:
    typedef std::shared_ptr<SimpleGPUScheduleObject> Ptr;
	enum class GPUPullFrontierType {
		BITMAP,
		BOOLMAP
	};

	enum class LoadBalancingModeType {
		VERTEX_BASED,
		TWC,
		TWCE,
		WM,
		CM,
		STRICT,
		EDGE_ONLY
	};

	enum class KernelFusionType {
		DISABLED,
		ENABLED
	};

	enum class DeduplicationType {
		ENABLED,
		DISABLED
	};

    enum class FrontierCreationType {
        FRONTIER_FUSED,
        UNFUSED_BITMAP,
        UNFUSED_BOOLMAP
    };

    enum class EdgeBlockingType {
        BLOCKED,
        UNBLOCKED
    };

    enum class DirectionType {
        DIR_PUSH,
        DIR_PULL
    };

private:
	LoadBalancingModeType load_balancing_mode;
	KernelFusionType kernel_fusion;
	DeduplicationType deduplication;
	abstract_schedule::FlexIntVal delta;
    FrontierCreationType frontier_creation;
    EdgeBlockingType edge_blocking;
    uint32_t edge_blocking_size;
    DirectionType direction;
    GPUPullFrontierType pull_frontier_type;

public:
	SimpleGPUScheduleObject() {
		load_balancing_mode = LoadBalancingModeType :: VERTEX_BASED;
		kernel_fusion = KernelFusionType :: DISABLED;
		deduplication = DeduplicationType ::DISABLED;
		direction = DirectionType ::DIR_PUSH;
		delta = abstract_schedule::FlexIntVal(1);
        edge_blocking_size = 0;
        frontier_creation = FrontierCreationType ::FRONTIER_FUSED;
        edge_blocking = EdgeBlockingType ::UNBLOCKED;
        pull_frontier_type = GPUPullFrontierType ::BOOLMAP;
	}

    SimpleGPUScheduleObject(SimpleGPUSchedule gpu_schedule) {
        configPullFrontierType(gpu_schedule.pull_frontier_rep);
        configKernelFusion(gpu_schedule.kernel_fusion);
        configDeduplicationType(gpu_schedule.deduplication);
        configDirection(gpu_schedule.direction);
        delta = abstract_schedule::FlexIntVal(gpu_schedule.delta);
        edge_blocking_size = gpu_schedule.edge_blocking_size;
        configFrontierCreationType(gpu_schedule.frontier_creation);
        configEdgeBlocking(gpu_schedule.edge_blocking);
        configLoadBalanceMode(gpu_schedule.load_balancing);
    }

	virtual SimpleScheduleObject::ParallelizationType getParallelizationType() override {
		switch(load_balancing_mode) {
			case SimpleGPUScheduleObject::LoadBalancingModeType ::VERTEX_BASED:
				return SimpleScheduleObject::ParallelizationType ::VERTEX_BASED;
			default:
				return SimpleScheduleObject::ParallelizationType ::EDGE_BASED;
		}
	}

    virtual SimpleScheduleObject::Direction getDirection() override {
	    switch(direction) {
	        case DirectionType ::DIR_PULL:
	            return SimpleScheduleObject::Direction ::PULL;
	        case DirectionType ::DIR_PUSH:
	            return SimpleScheduleObject::Direction ::PUSH;
	        default:
                assert(false && "Invalid option for GPU Direction\n");
                break;
	    }
    }

    virtual SimpleScheduleObject::Deduplication getDeduplication() override {
	    switch(deduplication) {
	        case DeduplicationType ::ENABLED:
	            return SimpleScheduleObject::Deduplication ::ENABLED;
	        case DeduplicationType ::DISABLED:
	            return SimpleScheduleObject::Deduplication ::DISABLED;
            default:
                assert(false && "Invalid option for GPU Deduplication\n");
                break;
	    }
    }

    virtual abstract_schedule::FlexIntVal getDelta() override {
        return delta;
    }

    virtual SimpleScheduleObject::PullFrontierType getPullFrontierType() override {
        switch(pull_frontier_type) {
            case GPUPullFrontierType ::BITMAP:
                return SimpleScheduleObject::PullFrontierType ::BITMAP;
            case GPUPullFrontierType ::BOOLMAP:
                return SimpleScheduleObject::PullFrontierType ::BOOLMAP;
            default:
                assert(false && "Invalid option for GPU Pull Frontier Type\n");
                break;
        }
    }

    SimpleGPUScheduleObject::DeduplicationType getGPUDeduplicationType() {
	    return deduplication;
	}

    SimpleGPUScheduleObject::KernelFusionType getKernelFusion() {
        return kernel_fusion;
    }

    SimpleGPUScheduleObject::LoadBalancingModeType getLoadBalancingMode() {
        return load_balancing_mode;
    }

    SimpleGPUScheduleObject::EdgeBlockingType getEdgeBlockingType() {
	    return edge_blocking;
	}

	SimpleGPUScheduleObject::FrontierCreationType getFrontierCreationType() {
	    return frontier_creation;
	}

	uint32_t getEdgeBlockingSize() {
	    return edge_blocking_size;
	}

    void configKernelFusion(KernelFusionType kernelFusionType) {
        kernel_fusion = kernelFusionType;
	}

    void configKernelFusion(SimpleGPUSchedule::kernel_fusion_type kernel_fusion_type) {
	    if (kernel_fusion_type == SimpleGPUSchedule::kernel_fusion_type::FUSION_DISABLED) {
            kernel_fusion = KernelFusionType ::DISABLED;
	    } else {
            kernel_fusion = KernelFusionType::ENABLED;
        }
    }

  void configDirection(DirectionType direction_type) {
      direction = direction_type;
  }

  void configDirection(SimpleGPUSchedule::direction_type direction_type) {
      if (direction_type == SimpleGPUSchedule::direction_type::DIR_PUSH) {
          direction = DirectionType ::DIR_PUSH;
      } else {
          direction = DirectionType ::DIR_PULL;
      }
  }

	void configEdgeBlocking(EdgeBlockingType edgeBlockingType) {
	    edge_blocking = edgeBlockingType;
	}

    void configEdgeBlocking(SimpleGPUSchedule::edge_blocking_type edgeBlockingType) {
        if (edgeBlockingType == SimpleGPUSchedule::edge_blocking_type::UNBLOCKED) {
            edge_blocking = EdgeBlockingType ::UNBLOCKED;
        } else {
            edge_blocking = EdgeBlockingType ::UNBLOCKED;
        }
    }

    void configDeduplicationType(DeduplicationType dedup_type) {
        deduplication = dedup_type;
	}

  void configDeduplicationType(SimpleGPUSchedule::deduplication_type dedup_type) {
        if (dedup_type == SimpleGPUSchedule::deduplication_type::DEDUP_ENABLED) {
            deduplication = DeduplicationType ::ENABLED;
        } else {
            deduplication = DeduplicationType::DISABLED;
        }
  }

  void configFrontierCreationType(FrontierCreationType frontier_type) {
      frontier_creation = frontier_type;
  }

  void configFrontierCreationType(SimpleGPUSchedule::frontier_creation_type frontier_type) {
      if (frontier_type == SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
          frontier_creation = FrontierCreationType ::FRONTIER_FUSED;
      } else if (frontier_type == SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
          frontier_creation = FrontierCreationType ::UNFUSED_BITMAP;
      } else {
          frontier_creation = FrontierCreationType ::UNFUSED_BOOLMAP;
      }
  }

  void configPullFrontierType(GPUPullFrontierType frontier_type) {
      pull_frontier_type = frontier_type;
  }

  void configPullFrontierType(SimpleGPUSchedule::pull_frontier_rep_type frontier_type) {
      if (frontier_type == SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
          pull_frontier_type = GPUPullFrontierType ::BITMAP;
      } else {
          pull_frontier_type = GPUPullFrontierType ::BOOLMAP;
      }
  }
  void configLoadBalanceMode(LoadBalancingModeType load_balancing_mode_type) {
      load_balancing_mode = load_balancing_mode_type;
  }

  void configLoadBalanceMode(SimpleGPUSchedule::load_balancing_type load_balancing_mode_type) {
      switch(load_balancing_mode_type) {
          case SimpleGPUSchedule::load_balancing_type ::VERTEX_BASED:
              load_balancing_mode = LoadBalancingModeType ::VERTEX_BASED;
              break;
          case SimpleGPUSchedule::load_balancing_type ::TWC:
              load_balancing_mode = LoadBalancingModeType ::TWC;
              break;
          case SimpleGPUSchedule::load_balancing_type ::TWCE:
              load_balancing_mode = LoadBalancingModeType ::TWCE;
              break;
          case SimpleGPUSchedule::load_balancing_type ::WM:
              load_balancing_mode = LoadBalancingModeType ::WM;
              break;
          case SimpleGPUSchedule::load_balancing_type ::CM:
              load_balancing_mode = LoadBalancingModeType ::CM;
              break;
          case SimpleGPUSchedule::load_balancing_type ::STRICT:
              load_balancing_mode = LoadBalancingModeType ::STRICT;
              break;
          case SimpleGPUSchedule::load_balancing_type ::EDGE_ONLY:
              load_balancing_mode = LoadBalancingModeType ::EDGE_ONLY;
              break;
          default:
              assert(false && "Invalid option for configLoadBalance");
              break;
      }
  }
};
    class HybridGPUScheduleObject : public abstract_schedule::CompositeScheduleObject {
    public:
        typedef std::shared_ptr<HybridGPUScheduleObject> Ptr;
        enum class CompositeCriteria {
            INPUT_VERTEXSET_SIZE
        };
    private:
        ScheduleObject::Ptr first_schedule;
        ScheduleObject::Ptr second_schedule;
        CompositeCriteria composite_criteria;
        float threshold;
    public:
        HybridGPUScheduleObject (ScheduleObject::Ptr first, ScheduleObject::Ptr second, float t, CompositeCriteria criteria=CompositeCriteria::INPUT_VERTEXSET_SIZE) {
            composite_criteria = criteria;
            threshold = t;
            first_schedule = first;
            second_schedule = second;
        }

      HybridGPUScheduleObject (HybridGPUSchedule hybrid_gpu_schedule) {
          if (hybrid_gpu_schedule._hybrid_criteria == HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE) {
              composite_criteria = CompositeCriteria ::INPUT_VERTEXSET_SIZE;
          }
          threshold = hybrid_gpu_schedule.threshold;
          first_schedule = std::make_shared<SimpleGPUScheduleObject>(hybrid_gpu_schedule.s1);
          second_schedule = std::make_shared<SimpleGPUScheduleObject>(hybrid_gpu_schedule.s2);
      }

        virtual ScheduleObject::Ptr getFirstScheduleObject() override {
            return first_schedule;
        }
        virtual ScheduleObject::Ptr getSecondScheduleObject() override {
            return second_schedule;
        }
        virtual CompositeCriteria getCompositeCriteria() {
            return composite_criteria;
        }
    };


}
}
}

#endif

