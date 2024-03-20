bool IsISPattern(const StmtPattern& pattern) {
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern) {
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern) {
  return std::holds_alternative<R>(pattern);
}


template <typename DoEachT>
void VisitStmtOpImpl(const IS& injective_source, const DoEachT& DoEach) {
  for (const auto* op : injective_source.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const PS& partial_shardable, const DoEachT& DoEach) {
  for (const auto* op : partial_shardable.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const R& reduce, const DoEachT& DoEach) {
  std::visit(adt::match{
                 [](const std::monostate&) {
                   // do nothing.
                 },
                 [&](const IS& injective_source) {
                   VisitStmtOpImpl(injective_source, DoEach);
                 },
                 [&](const PS& partial_shardable) {
                   VisitStmtOpImpl(partial_shardable, DoEach);
                 },
             },
             reduce.input);
  DoEach(reduce.reduce_op_pattern.reduce_op);
}

template <typename DoEachT>
void VisitStmtOp(const StmtPattern& stmt, const DoEachT& DoEach) {
  std::visit([&](const auto& impl) { VisitStmtOpImpl(impl, DoEach); }, stmt);
}

int GetOutputShardableAxesResultIdx(const pir::Operation* op) { return 0; }

pir::Value GetStmtBigestShapeValueImpl(const IS& injective_source) {
  const auto* sink_op = injective_source.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValueImpl(const R& reduce_pattern) {
  const auto* sink_op = reduce_pattern.reduce_op_pattern.reduce_op;
  CHECK_EQ(sink_op->num_operands(), 1);
  return sink_op->operand_source(0);
}

pir::Value GetStmtBigestShapeValueImpl(const PS& partial_shardable) {
  const auto* sink_op = partial_shardable.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValue(const StmtPattern& stmt) {
  return std::visit(
      [&](const auto& impl) { return GetStmtBigestShapeValueImpl(impl); },
      stmt);
}


const pir::Operation* GetStmtSoleSinkImpl(const IS& injective_source) {
  return injective_source.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const PS& partial_shardable) {
  return partial_shardable.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const R& reduce) {
  return reduce.reduce_op_pattern.reduce_op;
}

const pir::Operation* GetStmtSoleSinkOp(const StmtPattern& stmt) {
  return std::visit([](const auto& impl) { return GetStmtSoleSinkImpl(impl); },
                    stmt);
}


void SortStmtPtrs(
    std::vector<const StmtPattern*>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
  auto GetOrderValue4Stmt = [&](const StmtPattern* stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(*stmt);
    return OrderValue4Op(sink_op);
  };
  const auto Cmp = [&](const auto* lhs, const auto* rhs) {
    const auto& lhs_order = GetOrderValue4Stmt(lhs);
    const auto& rhs_order = GetOrderValue4Stmt(rhs);
    return lhs_order < rhs_order;
  };
  std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
}
common::TopoWalker<const StmtPattern*> MakeTopoWalker(
    const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns) {
using StmtPtrs = std::vector<const StmtPattern*>;
using Op2OwnerStmtPtrs =
    std::unordered_map<const pir::Operation*, StmtPtrs>;
auto op2owner_stmt_ptr = std::make_shared<Op2OwnerStmtPtrs>();
for (const auto& stmt : stmt_patterns) {
    VisitStmtOp(stmt, [&](const pir::Operation* op) {
    (*op2owner_stmt_ptr)[op].push_back(&stmt);
    });
}
using NodeVisitor = std::function<void(const StmtPattern*)>;
auto VisitInput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    VisitStmtOp(*stmt, [&](const auto* op) {
    op_topo.VisitInputOp(op, [&](const auto* input_op) {
        const auto& owners_iter = op2owner_stmt_ptr->find(input_op);
        if (owners_iter == op2owner_stmt_ptr->end()) return;
        if (owners_iter->second.size() != 1) return;
        const auto* owner_stmt = *owners_iter->second.begin();
        if (owner_stmt == stmt) return;
        DoEach(owner_stmt);
    });
    });
};
auto VisitOutput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    const auto* sink = GetStmtSoleSinkOp(*stmt);
    op_topo.VisitOutputOp(sink, [&](const pir::Operation* op) {
    const auto& owners_iter = op2owner_stmt_ptr->find(op);
    if (owners_iter == op2owner_stmt_ptr->end()) return;
    for (const StmtPattern* stmt : owners_iter->second) {
        DoEach(stmt);
    }
    });
};
const auto& TryPushBack = [](const auto* stmt, auto* stmts) {
    if (std::find(stmts->begin(), stmts->end(), stmt) == stmts->end()) {
    stmts->push_back(stmt);
    }
};
using EdgeCache =
    std::unordered_map<const StmtPattern*, std::vector<const StmtPattern*>>;
auto stmt2inputs = std::make_shared<EdgeCache>();
auto stmt2outputs = std::make_shared<EdgeCache>();
for (const auto& stmt : stmt_patterns) {
    (void)(*stmt2inputs)[&stmt];
    VisitInput(&stmt, [&](const auto* input) {
    TryPushBack(input, &(*stmt2inputs)[&stmt]);
    });
    (void)(*stmt2outputs)[&stmt];
    VisitOutput(&stmt, [&](const auto* output) {
    TryPushBack(output, &(*stmt2outputs)[&stmt]);
    });
}

auto VisitCachedInput = [stmt2inputs](const auto* stmt,
                                        const NodeVisitor& DoEach) {
    const auto& map = (*stmt2inputs);
    const auto& iter = map.find(stmt);
    if (iter == map.end()) return;
    for (const auto* input : iter->second) {
    DoEach(input);
    }
};
auto VisitCachedOutput = [stmt2outputs](const auto* stmt,
                                        const NodeVisitor& DoEach) {
    const auto& map = (*stmt2outputs);
    const auto& iter = map.find(stmt);
    if (iter == map.end()) return;
    for (const auto* output : iter->second) {
    DoEach(output);
    }
};
return common::TopoWalker<const StmtPattern*>(VisitCachedInput,
                                                VisitCachedOutput);

}

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const OpTopo& op_topo) {
  const auto& IsSource = [&](const pir::Operation* op) {
    std::size_t num_inputs = 0;
    op_topo.VisitInputOp(op,
                         [&](const pir::Operation* input) { ++num_inputs; });
    return num_inputs == 0;
  };

  const auto starts = [&] {
    std::list<const pir::Operation*> starts;
    for (const auto* op : *op_topo.ops) {
      if (IsSource(op)) {
        starts.push_back(op);
      } else {
        // do nothing.
      }
    }
    return starts;
  }();

  std::unordered_map<const pir::Operation*, bool> op_2_is_injective_source;

  auto IsInputsAllInjectiveSource = [&](const pir::Operation* op) {
    bool is_inputs_all_injective_source = true;
    op_topo.VisitInputOp(op, [&](const pir::Operation* input) {
      is_inputs_all_injective_source = (is_inputs_all_injective_source &&
                                        op_2_is_injective_source.at(input));
    });
    return is_inputs_all_injective_source;
  };
  const auto VisitInput = [&](const pir::Operation* op,
                              const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitOutput = [&](const pir::Operation* op,
                               const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> walker{VisitInput, VisitOutput};
  walker(starts.begin(), starts.end(), [&](const pir::Operation* op) {
    op_2_is_injective_source[op] =
        (IsGeneralInjective(op) && IsInputsAllInjectiveSource(op));
  });
  return [map = std::move(op_2_is_injective_source)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}