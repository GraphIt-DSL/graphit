#ifndef MIR_METADATA_H
#define MIR_METADATA_H

#include <memory>
#include <cassert>
namespace graphit {
namespace mir {

template<typename T>
class MIRMetadataImpl;

// The abstract class for the mir metadata
// Different templated metadata types inherit from this type
class MIRMetadata: public std::enable_shared_from_this<MIRMetadata> {
public:
	typedef std::shared_ptr<MIRMetadata> Ptr;
	virtual ~MIRMetadata() = default;


	template <typename T>
	bool isa (void) {
		if(std::dynamic_pointer_cast<MIRMetadataImpl<T>>(shared_from_this()))
			return true;
		return false;
	}
	template <typename T>
	std::shared_ptr<MIRMetadataImpl<T>> to(void) {
		std::shared_ptr<MIRMetadataImpl<T>> ret = std::dynamic_pointer_cast<MIRMetadataImpl<T>>(shared_from_this());
		assert(ret != nullptr);
		return ret;
	}
};

// Templated metadata class for each type
template<typename T>
class MIRMetadataImpl: public MIRMetadata {
public:
	typedef std::shared_ptr<MIRMetadataImpl<T>> Ptr;
	T val;	
	MIRMetadataImpl(T _val): val(_val) {
	}
};

}
}
#endif
