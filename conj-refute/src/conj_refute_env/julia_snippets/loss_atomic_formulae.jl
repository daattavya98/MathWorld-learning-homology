using DynamicExpressions
using LossFunctions
using Statistics
using SymbolicRegression: Dataset
using Logging

# LOG_PATH = joinpath(@__DIR__, "atomic_formulae_loss.log")

# #1)Open log file
# _log_io = open(LOG_PATH, "w")

# #2)Define a custom logger for our specific use (don't set it globally)
# custom_logger = SimpleLogger(_log_io, Logging.Info)
# atexit(() -> close(_log_io))

struct SignumLossAtomicFormulae <: Function
  complexityWeight::Real
  equalsUsedBoost::Real
  plusUsedBoost::Real
  timesUsedBoost::Real
  minusUsedBoost::Real
  outerEqualsPenalty::Real
  equalsSemanticPenalty::Real
  dataErrorPenalty::Real
  weights::Vector{Float64}
end

# Test-only toggle (controlled from Python script):
# when true, treat feature index 8 (nullity_D2) as constant 1 in symbolic simplification.
replace_nullity_d2_with_one = false
"""
    SignumLossAtomicFormulae(
      ; complexityWeight::Real,
        equalsUsedBoost::Real,
        plusUsedBoost::Real,
        timesUsedBoost::Real,
        minusUsedBoost::Real,
        outerEqualsPenalty::Real,
        equalsSemanticPenalty::Real,
        dataErrorPenalty::Real,
        weights::Vector{Float64}
    )
"""

function SignumLossAtomicFormulae(
    ; complexityWeight::Real,
      equalsUsedBoost::Real,
      plusUsedBoost::Real,
      timesUsedBoost::Real,
      minusUsedBoost::Real,
      outerEqualsPenalty::Real,
      equalsSemanticPenalty::Real,
      dataErrorPenalty::Real,
      weights::Vector{Float64}
    )
    # Internally call the positional constructor:
    return SignumLossAtomicFormulae(
      complexityWeight,
      equalsUsedBoost,
      plusUsedBoost,
      timesUsedBoost,
      minusUsedBoost,
      outerEqualsPenalty,
      equalsSemanticPenalty,
      dataErrorPenalty,
      weights
    )
end

"""
    make_atomic_loss_structure(
      complexityWeight::Real,
      equalsUsedBoost::Real,
      plusUsedBoost::Real,
      timesUsedBoost::Real,
      minusUsedBoost::Real,
      outerEqualsPenalty::Real,
      equalsSemanticPenalty::Real,
      dataErrorPenalty::Real
    ) -> SignumLossAtomicFormulae
Construct a new `SignumLossAtomicFormulae` using the global `data_weights` as its `weights` field.
"""
function make_atomic_loss_structure(
    complexityWeight::Real,
    equalsUsedBoost::Real,
    plusUsedBoost::Real,
    timesUsedBoost::Real,
    minusUsedBoost::Real,
    outerEqualsPenalty::Real,
    equalsSemanticPenalty::Real,
    dataErrorPenalty::Real
  )::SignumLossAtomicFormulae
  return SignumLossAtomicFormulae(
    complexityWeight=complexityWeight,
    equalsUsedBoost=equalsUsedBoost,
    plusUsedBoost=plusUsedBoost,
    timesUsedBoost=timesUsedBoost,
    minusUsedBoost=minusUsedBoost,
    outerEqualsPenalty=outerEqualsPenalty,
    equalsSemanticPenalty=equalsSemanticPenalty,
    dataErrorPenalty=dataErrorPenalty,
    weights=Array{Float64}(data_weights)
  )
end

"""
    exponential_decay(x0, γ; n_steps)

Return an array of length `n_steps+1` with `x_t = x0 * γ^t`.
"""
function exponential_decay(x0, γ; n_steps=100)
    if n_steps == 0
        return 0.0
    end
    vals = x0 .* (γ .^ (0:n_steps))
    sum_vals = sum(vals)
    return sum_vals
end


# Helper function to check if a tree contains any equals operations
function contains_equals_operation(node)
    if node.degree == 0
        # Leaf node - no equals operation
        return false
    elseif node.degree == 2
        # Check if this node is an equals operation
        if node.op == 4  # equals_index
            return true
        end
        # Recursively check children
        return contains_equals_operation(node.l) || contains_equals_operation(node.r)
    else
        return false
    end
end

# Helper function to build simplified expressions from tree nodes
function build_simplified_expression(node)
    # Operation indices
    plus_index = 1
    times_index = 2
    minus_index = 3
    equals_index = 4
    
    # Base case: leaf node (feature)
    if node.degree == 0
        if replace_nullity_d2_with_one && node.feature == 8  # nullity_D2
            return 1
        elseif node.feature == 9  # _0 is feature 9 (1-based indexing)
            return 0
        elseif node.feature == 10  # _1 is feature 10 (1-based indexing)
            return 1
        elseif node.feature == 11  # _2 is feature 11 (1-based indexing)
            return 2
        else
            return Symbol("x$(node.feature)")  # Return as symbol like :x1, :x2, etc.
        end
    end
    
    # Binary operations
    if node.degree == 2
        left_expr = build_simplified_expression(node.l)
        right_expr = build_simplified_expression(node.r)
        
        if node.op == plus_index
            return simplify_addition(left_expr, right_expr)
        elseif node.op == times_index
            return simplify_multiplication(left_expr, right_expr)
        elseif node.op == minus_index
            return simplify_subtraction(left_expr, right_expr)
        else
            # For any other operation (including equals), return a generic representation
            return (Symbol("op$(node.op)"), left_expr, right_expr)
        end
    end
    
    return nothing  # Fallback for unexpected cases
end

# Helper functions for algebraic simplification
function simplify_addition(left, right)
    # Handle numeric constants
    if isa(left, Number) && isa(right, Number)
        return left + right
    end
    
    # Handle zero cases
    if left == 0
        return right
    elseif right == 0
        return left
    end
    
    # Sort operands for canonical form (symbols come first, then expressions)
    if should_sort_first(left, right)
        return (:+, left, right)
    else
        return (:+, right, left)
    end
end

function simplify_multiplication(left, right)
    # Handle numeric constants
    if isa(left, Number) && isa(right, Number)
        return left * right
    end
    
    # Handle zero cases
    if left == 0 || right == 0
        return 0
    end
    
    # Handle one cases
    if left == 1
        return right
    elseif right == 1
        return left
    end
    
    # Sort operands for canonical form
    if should_sort_first(left, right)
        return (:*, left, right)
    else
        return (:*, right, left)
    end
end

function simplify_subtraction(left, right)
    # Handle numeric constants
    if isa(left, Number) && isa(right, Number)
        return left - right
    end
    
    # Handle zero cases
    if right == 0
        return left
    elseif left == 0
        return -right  # This could be represented as (:-, 0, right) or simplified further
    end
    
    # Handle same operands
    if left == right
        return 0
    end
    
    return (:-, left, right)
end

# Helper function to determine canonical ordering for expressions
function should_sort_first(a, b)
    # Numbers come after symbols
    if isa(a, Number) && !isa(b, Number)
        return false
    elseif !isa(a, Number) && isa(b, Number)
        return true
    end
    
    # If both are symbols, sort alphabetically
    if isa(a, Symbol) && isa(b, Symbol)
        return string(a) <= string(b)
    end
    
    # If both are expressions (tuples), sort by operation first, then recursively
    if isa(a, Tuple) && isa(b, Tuple)
        if length(a) > 0 && length(b) > 0
            if a[1] != b[1]
                return string(a[1]) <= string(b[1])
            else
                # Same operation, compare recursively
                return string(a) <= string(b)
            end
        end
    end
    
    # Default case: convert to string and compare
    return string(a) <= string(b)
end

function equals_semantic_check(left_child, right_child)
    # First, check if either child (or their descendants) contains any equals operations
    # If so, this is an invalid structure and we return false immediately
    if contains_equals_operation(left_child) || contains_equals_operation(right_child)
        return false
    end
    
    try
        # Build simplified expressions for both children
        left_expr = build_simplified_expression(left_child)
        right_expr = build_simplified_expression(right_child)
        
        # Compare expressions exactly
        return left_expr == right_expr
        
    catch e
        # Fallback to structural comparison if symbolic comparison fails
        # @debug "Symbolic comparison failed, falling back to structural comparison: $e"
        return tree_structural_equal(left_child, right_child)
    end
end

# Helper function for structural equality checking
function tree_structural_equal(node1, node2)
    # Check if both nodes have the same degree
    if node1.degree != node2.degree
        return false
    end
    
    # For leaf nodes
    if node1.degree == 0
        return node1.feature == node2.feature
    end
    
    # For binary nodes
    if node1.degree == 2
        return (node1.op == node2.op && 
                tree_structural_equal(node1.l, node2.l) && 
                tree_structural_equal(node1.r, node2.r))
    end
    
    return false
end


function (loss::SignumLossAtomicFormulae)(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
  # Periodic cache clearing at the start of each loss evaluation
  
  X = copy(dataset.X)
  y = copy(dataset.y)
  if idx !== nothing  # Batching support
    X = X[:, idx]
    y = y[idx]
  end

  complexity = sum(1 for _ in tree)

  plus_index = 1
  times_index = 2
  minus_index = 3
  equals_index = 4

  plus_used_count = 0
  times_used_count = 0
  minus_used_count = 0
  equals_used_count = 0

  equals_semantic_flag = false

  decay_param_plus = 0.4
  decay_param_times = 0.4
  decay_param_minus = 0.4
  decay_param_equals = 0.4
  decay_param_complexity = 0.4

  for node in tree
    is_plus_node = node.degree == 2 && node.op == plus_index
    is_times_node = node.degree == 2 && node.op == times_index
    is_minus_node = node.degree == 2 && node.op == minus_index
    is_equals_node = node.degree == 2 && node.op == equals_index

    if is_plus_node
        left_child = node.l
        right_child = node.r
        if (left_child.degree == 2 && left_child.op == equals_index) || (right_child.degree == 2 && right_child.op == equals_index)
            return L(Inf)  # Invalid tree structure
        end
        plus_used_count += 1
    end
    if is_times_node
        left_child = node.l
        right_child = node.r
        if (left_child.degree == 2 && left_child.op == equals_index) || (right_child.degree == 2 && right_child.op == equals_index)
            return L(Inf)  # Invalid tree structure
        end
      times_used_count += 1
    end
    if is_minus_node
        left_child = node.l
        right_child = node.r
        if (left_child.degree == 2 && left_child.op == equals_index) || (right_child.degree == 2 && right_child.op == equals_index)
            return L(Inf)  # Invalid tree structure
        end
        minus_used_count += 1
    end
    if is_equals_node
        left_child = node.l
        right_child = node.r
        if (left_child.degree == 2 && left_child.op == equals_index) || (right_child.degree == 2 && right_child.op == equals_index)
            return L(Inf)  # Invalid tree structure
        end
        equals_used_count += 1
        # Check for semantic errors in equals nodes
        equals_semantic_flag = equals_semantic_check(left_child, right_child)
    end
  end

  # Check if the root node is an equals node
  if (tree.degree == 2 && tree.op == equals_index)
    outerPenaltyFlag = false
  else
    outerPenaltyFlag = true
  end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end

  correct = (y_pred .== y)
  W = loss.weights
  if length(W) == 0
    Wvals = ones(length(correct))
  else
    Wvals = W
  end
  
  # Ensure weights are positive to prevent negative weighted accuracy
  if length(Wvals) != length(correct)
    Wvals = ones(length(correct))  # Fallback to uniform weights if size mismatch
  else
    Wvals = abs.(Wvals)  # Make all weights positive
    if sum(Wvals) == 0.0
      Wvals = ones(length(correct))  # Fallback if all weights were zero
    end
  end
  
  # Debug logging for weighted accuracy issues
  num_correct = sum(correct)
  total_data = length(correct)
  sum_weights_correct = sum(Wvals[correct])
  sum_all_weights = sum(Wvals)
  
  # Weighted mean of "correctness"
  weighted_acc = sum(Wvals[correct]) / sum(Wvals)
  
  # Log debug info when we have issues
  if weighted_acc > 1.0 || weighted_acc < 0.0
      # with_logger(custom_logger) do
      #   @info "WEIGHTED_ACC_DEBUG: weighted_acc=$weighted_acc, num_correct=$num_correct, total_data=$total_data, sum_weights_correct=$sum_weights_correct, sum_all_weights=$sum_all_weights, Wvals_length=$(length(Wvals)), Wvals_sample=$(Wvals[1:min(5,length(Wvals))]), y_pred_sample=$(y_pred[1:min(5,length(y_pred))]), y_sample=$(y[1:min(5,length(y))])"
      # end
      # flush(_log_io)
  end

  complexity_decay_val = exponential_decay(loss.complexityWeight, decay_param_complexity; n_steps=complexity)
  plus_used_decay_val = exponential_decay(loss.plusUsedBoost, decay_param_plus; n_steps=plus_used_count)
  times_used_decay_val = exponential_decay(loss.timesUsedBoost, decay_param_times; n_steps=times_used_count)
  minus_used_decay_val = exponential_decay(loss.minusUsedBoost, decay_param_minus; n_steps=minus_used_count)
  equals_used_decay_val = exponential_decay(loss.equalsUsedBoost, decay_param_equals; n_steps=equals_used_count)

  if outerPenaltyFlag
    outer_penalty = loss.outerEqualsPenalty
  else
    outer_penalty = 0.0
  end

  exponent_sum = (loss.dataErrorPenalty * (1.0 - weighted_acc)) - complexity_decay_val - plus_used_decay_val - times_used_decay_val - minus_used_decay_val - equals_used_decay_val + outer_penalty
  if equals_semantic_flag
    exponent_sum += loss.equalsSemanticPenalty
  end

  signumLoss = exp(exponent_sum)

  # Debug print for exact zero losses with critical issue detection
  if signumLoss == 0.0 || signumLoss < 1e-60
    # with_logger(custom_logger) do
    #   @info "DEBUG: Zero/tiny loss detected! signumLoss: $signumLoss, tree: $tree, complexity: $complexity, exponent_sum: $exponent_sum, weighted_acc: $weighted_acc, plus_decay: $plus_used_decay_val, times_decay: $times_used_decay_val, minus_decay: $minus_used_decay_val, equals_decay: $equals_used_decay_val"
    # end
    # flush(_log_io)
  end
  return L(signumLoss)

end