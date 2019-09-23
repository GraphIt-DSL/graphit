#ifndef EXTRACT_READ_WRITE_H
#define EXTRACT_READ_WRITE_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
namespace graphit {
class ExtractReadWriteSet: public mir::MIRVisitor {
public:
	ExtractReadWriteSet(MIRContext *mir_context_): read_set(read_set_), write_set(write_set_), mir_context(mir_context_) {
	}
	const std::vector<mir::TensorArrayReadExpr::Ptr> &read_set;
	const std::vector<mir::TensorArrayReadExpr::Ptr> &write_set;	
	
protected:
	virtual void visit(mir::TensorArrayReadExpr::Ptr);
	virtual void visit(mir::AssignStmt::Ptr);
	virtual void visit(mir::StmtBlock::Ptr);	
	MIRContext *mir_context;
	
private:
	void add_read(mir::TensorArrayReadExpr::Ptr);
	void add_write(mir::TensorArrayReadExpr::Ptr);
	
	std::vector<mir::TensorArrayReadExpr::Ptr> read_set_;
	std::vector<mir::TensorArrayReadExpr::Ptr> write_set_;
};
}

#endif
