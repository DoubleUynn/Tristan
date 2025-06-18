import torch
import os
import utils.config as cfg
from utils.genetic_algorithm import Brain, mutation, device  # Import device from genetic_algorithm
import numpy as np

def compare_models(model1, model2):
    total_params = 0
    different_params = 0
    differences = []
    
    # Compare each parameter tensor
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if param1.requires_grad:
            # Calculate absolute differences
            diff = torch.abs(param1 - param2)
            # Count parameters that changed
            changed = (diff > 0).sum().item()
            different_params += changed
            total_params += param1.numel()
            
            # Record differences for statistics
            if changed > 0:
                mean_diff = torch.mean(diff[diff > 0]).item()
                max_diff = torch.max(diff).item()
                differences.append((name1, changed, param1.numel(), mean_diff, max_diff))
    
    # Calculate overall statistics
    difference_percentage = (different_params / total_params) * 100 if total_params > 0 else 0
    
    # Calculate max and average differences across all parameters
    max_difference = max([d[4] for d in differences]) if differences else 0
    avg_difference = sum([d[3] for d in differences]) / len(differences) if differences else 0
    
    return difference_percentage, max_difference, avg_difference, differences

def verify_mutation(generation=0, num_tests=5):
    print(f"Verifying mutation function with {num_tests} test models (generation {generation})")
    print("-" * 80)
    
    # Statistics collectors
    all_diff_percentages = []
    all_max_diffs = []
    all_avg_diffs = []
    
    for test_num in range(num_tests):
        # Create a model
        original_model = Brain().to(device)
        
        # Create a deep copy of the model
        copied_model = Brain().to(device)
        copied_model.load_state_dict(original_model.state_dict())
        
        # Verify the copy is identical
        diff_pct, max_diff, avg_diff, _ = compare_models(original_model, copied_model)
        if diff_pct > 0:
            print(f"WARNING: Copy verification failed! Models differ by {diff_pct:.6f}%")
        
        # Apply mutation to the copied model
        print(f"\nTest {test_num+1}: Applying mutation (generation {generation})...")
        mutated_model = mutation(copied_model, generation)
        
        # Compare original and mutated models
        diff_pct, max_diff, avg_diff, differences = compare_models(original_model, mutated_model)
        
        # Collect statistics
        all_diff_percentages.append(diff_pct)
        all_max_diffs.append(max_diff)
        all_avg_diffs.append(avg_diff)
        
        # Report results
        print(f"Mutation changed {diff_pct:.3f}% of parameters")
        print(f"Maximum parameter change: {max_diff:.6f}")
        print(f"Average parameter change: {avg_diff:.6f}")
        
        # Show detailed breakdown of changes by layer
        print("\nChanges by layer:")
        print(f"{'Layer':<30} {'Changed':<10} {'Total':<10} {'%Changed':<10} {'Avg Diff':<10} {'Max Diff':<10}")
        print("-" * 80)
        for name, changed, total, mean_diff, max_diff in differences:
            pct = (changed / total) * 100 if total > 0 else 0
            print(f"{name:<30} {changed:<10} {total:<10} {pct:<10.2f} {mean_diff:<10.6f} {max_diff:<10.6f}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL MUTATION STATISTICS:")
    print(f"Average percentage of parameters changed: {np.mean(all_diff_percentages):.3f}%")
    print(f"Average maximum parameter change: {np.mean(all_max_diffs):.6f}")
    print(f"Average of average parameter changes: {np.mean(all_avg_diffs):.6f}")
    
    # Conclusion
    if np.mean(all_diff_percentages) < 0.1:
        print("\nCRITICAL ISSUE: Mutation is changing very few parameters (<0.1%)")
    elif np.mean(all_diff_percentages) < 1.0:
        print("\nWARNING: Mutation is changing fewer parameters than expected (<1%)")
    else:
        print("\nMutation appears to be working correctly")
    
    if np.mean(all_max_diffs) < 0.0001:
        print("CRITICAL ISSUE: Maximum parameter changes are extremely small")
    
    # Calculate expected mutation rate based on cfg.MUTATION_FREQUENCY
    expected_rate = cfg.MUTATION_FREQUENCY / 100.0
    actual_rate = np.mean(all_diff_percentages) / 100.0
    
    print(f"\nExpected mutation rate: ~{expected_rate:.1%}")
    print(f"Actual mutation rate: {actual_rate:.3%}")
    
    if abs(expected_rate - actual_rate) > 0.1:
        print(f"ISSUE: Actual mutation rate differs significantly from expected rate")

if __name__ == "__main__":
    # Test with different generation values
    print("\n\nTESTING EARLY GENERATION MUTATION (generation < 100)")
    verify_mutation(generation=10, num_tests=3)
    
    print("\n\nTESTING MID GENERATION MUTATION (100 <= generation < 300)")
    verify_mutation(generation=150, num_tests=3)
    
    print("\n\nTESTING LATE GENERATION MUTATION (generation >= 300)")
    verify_mutation(generation=350, num_tests=3)
