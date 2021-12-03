//
// Created by Ajay Brahmakshatriya 
//

#ifndef GRAPHIT_GPU_SCHEDULE
#define GRAPHIT_GPU_SCHEDULE

#include <assert.h>


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


}
}
}

#endif

