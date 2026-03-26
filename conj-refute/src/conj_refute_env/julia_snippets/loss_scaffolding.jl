using DynamicExpressions
using LossFunctions
using Statistics
using SymbolicRegression: Dataset
using Logging

# LOG_PATH = joinpath(@__DIR__, "signum_loss.log")

# #1)Open log file
# _log_io = open(LOG_PATH, "w")

# #2)Define a custom logger for our specific use (don't set it globally)
# custom_logger = SimpleLogger(_log_io, Logging.Info)
# atexit(() -> close(_log_io))

struct ScaffoldingSignumLoss <: Function
  complexityWeight::Real
  logicalNegUsedBoost::Real
  impliesUsedBoost::Real
  conjunctionUsedBoost::Real
  outerNodePenalty::Real
  impliesSemanticErrorPenalty::Real
  logicalFalsePenalty::Real
  dataErrorPenalty::Real
  repeatPenalty::Real
  uniformPenalty::Real
  weights::Vector{Float64}
  atomicBlockLength::Real
end

"""
    ScaffoldingSignumLoss(
      ; complexityWeight::Real,
        logicalNegUsedBoost::Real,
        impliesUsedBoost::Real,
        conjunctionUsedBoost::Real,
        outerNodePenalty::Real,
        impliesSemanticErrorPenalty::Real,
        logicalFalsePenalty::Real,
        dataErrorPenalty::Real,
        repeatPenalty::Real,
        uniformPenalty::Real,
        weights::Vector{Float64},
        atomicBlockLength::Real
    )
Create a new `ScaffoldingSignumLoss` using keyword‐argument style. 
"""
function ScaffoldingSignumLoss(
    ; complexityWeight::Real,
      logicalNegUsedBoost::Real,
      impliesUsedBoost::Real,
      conjunctionUsedBoost::Real,
      outerNodePenalty::Real,
      impliesSemanticErrorPenalty::Real,
      logicalFalsePenalty::Real,
      dataErrorPenalty::Real,
      repeatPenalty::Real,
      uniformPenalty::Real,
      weights::Vector{Float64},
      atomicBlockLength::Real
    )
    # Internally call the positional constructor:
    return ScaffoldingSignumLoss(
      complexityWeight,
      logicalNegUsedBoost,
      impliesUsedBoost,
      conjunctionUsedBoost,
      outerNodePenalty,
      impliesSemanticErrorPenalty,
      logicalFalsePenalty,
      dataErrorPenalty,
      repeatPenalty,
      uniformPenalty,
      weights,
      atomicBlockLength
    )
end

"""
    make_scaffolding_loss_structure(
      complexityWeight::Real,
      logicalNegUsedBoost::Real,
      impliesUsedBoost::Real,
      conjunctionUsedBoost::Real,
      outerNodePenalty::Real,
      impliesSemanticErrorPenalty::Real,
      logicalFalsePenalty::Real,
      repeatPenalty::Real,
      uniformPenalty::Real,
      dataErrorPenalty::Real,
      atomicBlockLength::Real
    ) -> ScaffoldingSignumLoss
Construct a new `ScaffoldingSignumLoss` using the global `ref_weights` as its `weights` field.
"""
function make_scaffolding_loss_structure(
    complexityWeight::Real,
    logicalNegUsedBoost::Real,
    impliesUsedBoost::Real,
    conjunctionUsedBoost::Real,
    outerNodePenalty::Real,
    impliesSemanticErrorPenalty::Real,
    logicalFalsePenalty::Real,
    repeatPenalty::Real,
    uniformPenalty::Real,
    dataErrorPenalty::Real,
    atomicBlockLength::Real
  )::ScaffoldingSignumLoss
  return ScaffoldingSignumLoss(
    complexityWeight=complexityWeight,
    logicalNegUsedBoost=logicalNegUsedBoost,
    impliesUsedBoost=impliesUsedBoost,
    conjunctionUsedBoost=conjunctionUsedBoost,
    outerNodePenalty=outerNodePenalty,
    impliesSemanticErrorPenalty=impliesSemanticErrorPenalty,
    logicalFalsePenalty=logicalFalsePenalty,
    repeatPenalty=repeatPenalty,
    uniformPenalty=uniformPenalty,
    dataErrorPenalty=dataErrorPenalty,
    weights=Array{Float64}(ref_weights),
    atomicBlockLength=atomicBlockLength
  )
end

"""
    count_arithmetic_in_subtree(node, plus_index, times_index, minus_index)

Recursively count all arithmetic operators (+, *, -) in a subtree.
Returns the total count of arithmetic operators found.
"""
# function count_arithmetic_in_subtree(node, plus_index, times_index, minus_index)
#     count = 0
    
#     # Check if current node is an arithmetic operator
#     if node.degree == 2 && (node.op == plus_index || node.op == times_index || node.op == minus_index)
#         count += 1
#     end
    
#     # Recursively check children
#     if node.degree >= 1  # Has left child
#         count += count_arithmetic_in_subtree(node.l, plus_index, times_index, minus_index)
#     end
#     if node.degree == 2  # Has right child
#         count += count_arithmetic_in_subtree(node.r, plus_index, times_index, minus_index)
#     end
    
#     return count
# end

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

"""
  Count repeats of variables in the tree
"""
function count_repeats_in_tree(tree)
  var_counts = Dict{Int, Int}()
  
  for node in tree
    if node.degree == 0  # Leaf node
      if node.constant == false
        var_index = node.feature
        if haskey(var_counts, var_index)
          var_counts[var_index] += 1
        else
          var_counts[var_index] = 1
        end
      end
    end
  end
  
  repeat_count = 0
  for count in values(var_counts)
    if count > 1
      repeat_count += count - 1
    end
  end
  return repeat_count
end

"""
  Count uniform usage of variables in the tree
"""
function count_uniform_in_tree(tree, X, y)
  checked_vars = Set{Int}()
  uniform_count = 0

  for node in tree
    if node.degree == 0 && node.constant == false
      var_index = node.feature
      if !(var_index in checked_vars)
        feature_column = X[var_index, :]
        # I need to look at the union vector, y, and see if the feature column is all 0s or all 1s over this y. So basically, over all the columns that are non-zero in y, I need to see if the feature column is uniform.
        nonzero_indices = findall(!iszero, y)
        feature_subset = feature_column[nonzero_indices]
        is_feature_uniform = all(feature_subset .== 0) || all(feature_subset .== 1)
        if is_feature_uniform
          uniform_count += 1
        end
        # println("Variable index: $var_index, is uniform: $is_feature_uniform")
        push!(checked_vars, var_index)
      end
    end
  end
  return uniform_count
end

function (loss::ScaffoldingSignumLoss)(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
  X = copy(dataset.X)
  y = copy(dataset.y)
  if idx !== nothing  # Batching support
    X = X[:, idx]
    y = y[idx]
  end

  complexity = sum((1 for node in tree if !(node.degree == 0 && node.constant == false)); init=0) + loss.atomicBlockLength

  # Require that `implies` always acts on an atomic formula. Both left and right children must either have `logical_neg` as an operator.
  # Require that `logical_neg` always acts on an atomic formula.
  # Require that the arithmetic operators act correctly.

  implies_index = 1
  conjunction_index = 2
  logical_neg_index = 1
  
  logical_neg_used_count = 0
  implies_used_count = 0
  conjunction_used_count = 0
  implies_semantic_error_count = 0

  # Count occurrences of repeats of variables in the tree
  repeat_count = count_repeats_in_tree(tree)
  uniform_count = count_uniform_in_tree(tree, X, y)
  # @show uniform_count

  decay_param_implies = 0.4
  decay_param_conjunction = 0.4
  decay_param_logical_neg = 0.4
  decay_param_complexity = 0.4

  flag = 0

  for node in tree
    is_implies_node = node.degree == 2 && node.op == implies_index
    is_neg_node = node.degree == 1 && node.op == logical_neg_index
    is_conjunction_node = node.degree == 2 && node.op == conjunction_index

    if is_implies_node
      left_child = node.l
      right_child = node.r
      implies_used_count += 1

    end

    # Trying to avoid logical negation syntax errors entirely
    if is_neg_node
      logical_neg_used_count += 1
      child = node.l 
      if child.degree == 1 && child.op == logical_neg_index
        # This is not a valid logical negation
        return L(Inf)
      end
    end

    if is_conjunction_node
      left_child = node.l
      right_child = node.r
      conjunction_used_count += 1
    end
  end

  outerNodeImplies = false
  outerNodeConjunction = false
  outerNodeLogicalNeg = false
  if (tree.degree == 1 && tree.op == logical_neg_index)
    outerNodeLogicalNeg = true
  elseif (tree.degree == 2 && tree.op == implies_index)
    outerNodeImplies = true
  elseif (tree.degree == 2 && tree.op == conjunction_index)
    outerNodeConjunction = true
  end

  if outerNodeImplies
    outer_penalty = 0
  elseif outerNodeConjunction
    outer_penalty = 0.5 * loss.outerNodePenalty
  elseif outerNodeLogicalNeg
    outer_penalty = 0.75 * loss.outerNodePenalty
  else
    outer_penalty = loss.outerNodePenalty
  end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end

  # y_pred only tells me whether the overall logical statement is true or false for each data point
  # The other data penalty I am interested in the penalty if the scaffold ga does not use block atomic formulas that have better data performance

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
  
  # Weighted mean of "correctness"
  weighted_acc = sum(Wvals[correct]) / sum(Wvals)

  # Repeat data error computation basically for each row of the dataset
  weighted_acc_blk = 0.0

  complexity_decay_val = exponential_decay(loss.complexityWeight, decay_param_complexity; n_steps=complexity)
  logical_neg_used_decay_val = exponential_decay(loss.logicalNegUsedBoost, decay_param_logical_neg; n_steps=logical_neg_used_count)
  implies_used_decay_val = exponential_decay(loss.impliesUsedBoost, decay_param_implies; n_steps=implies_used_count)
  conjunction_used_decay_val = exponential_decay(loss.conjunctionUsedBoost, decay_param_conjunction; n_steps=conjunction_used_count)

  exponent_sum = (loss.logicalFalsePenalty * (1.0 - weighted_acc)) - complexity_decay_val - logical_neg_used_decay_val - implies_used_decay_val - conjunction_used_decay_val + loss.repeatPenalty * repeat_count + loss.uniformPenalty * uniform_count
  signumLoss = exp(exponent_sum)
  
  return L(signumLoss)
end
