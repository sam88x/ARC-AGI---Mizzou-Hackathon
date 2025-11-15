import json
import numpy as np
from typing import Tuple, Dict
import argparse
import sys

def score_submission_enhanced(submission_file_name, solutions_file_name, include_task_scores=False) -> dict:
    """
    Score a submission against ground truth solutions with both exact match and pixel accuracy.

    Args:
        submission_file_name (str): The file name of the submission file.
        solutions_file_name (str): The file name of the ground truth solutions.
        include_task_scores (bool, optional): Whether to include individual task scores. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - total_exact_match_score: Traditional exact match score
            - total_pixel_accuracy: Average pixel-level accuracy across all tasks
            - total_tasks_scored: Number of tasks evaluated
            - task_scores: Individual task scores (if requested)
    """
    # Open submission & solutions files
    with open(submission_file_name, "r") as file:
        submission = json.load(file)
    
    with open(solutions_file_name, "r") as file:
        solutions = json.load(file)

    total_exact_match_score = 0
    total_pixel_correct = 0
    total_pixel_count = 0
    total_tasks = 0
    task_scores = {}

    # Loop through each task
    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_exact_match = 0
        task_pixel_correct = 0
        task_pixel_total = 0
        num_pairs = len(task_submission)

        # Go through each task pair
        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = False
            ground_truth = solutions[task_id][pair_index]
            
            # Track best pixel accuracy for this pair across both attempts
            best_pixel_accuracy = 0
            
            # Look at both attempts
            for attempt_key, attempt in pair_attempts.items():
                
                # Check exact match
                if attempt == ground_truth:
                    pair_correct = True
                
                # Calculate pixel accuracy
                pixel_accuracy = calculate_pixel_accuracy(attempt, ground_truth)
                best_pixel_accuracy = max(best_pixel_accuracy, pixel_accuracy)
            
            # Record exact match
            if pair_correct:
                task_exact_match += 1
            
            # Accumulate pixel statistics
            gt_height = len(ground_truth)
            gt_width = len(ground_truth[0]) if gt_height > 0 else 0
            num_pixels = gt_height * gt_width
            
            task_pixel_correct += best_pixel_accuracy * num_pixels
            task_pixel_total += num_pixels

        # Calculate task-level scores
        task_exact_match_score = task_exact_match / num_pairs
        task_pixel_accuracy = task_pixel_correct / task_pixel_total if task_pixel_total > 0 else 0

        # Add to totals
        total_exact_match_score += task_exact_match_score
        total_pixel_correct += task_pixel_correct
        total_pixel_count += task_pixel_total

        # Log task scores
        task_scores[task_id] = {
            'exact_match': task_exact_match_score,
            'pixel_accuracy': task_pixel_accuracy,
            'pixels_correct': int(task_pixel_correct),
            'total_pixels': task_pixel_total
        }

    # Calculate overall metrics
    overall_pixel_accuracy = total_pixel_correct / total_pixel_count if total_pixel_count > 0 else 0

    result = {
        'total_exact_match_score': total_exact_match_score,
        'total_pixel_accuracy': overall_pixel_accuracy,
        'total_tasks_scored': total_tasks,
        'pixels_correct': int(total_pixel_correct),
        'total_pixels': total_pixel_count,
        'exact_match_percentage': (total_exact_match_score / total_tasks * 100) if total_tasks > 0 else 0,
        'pixel_accuracy_percentage': overall_pixel_accuracy * 100
    }

    if include_task_scores:
        result['task_scores'] = task_scores

    return result


def calculate_pixel_accuracy(prediction, ground_truth):
    """
    Calculate pixel-level accuracy between prediction and ground truth.
    Handles different grid sizes by only comparing overlapping regions.
    
    Args:
        prediction: 2D list representing predicted grid
        ground_truth: 2D list representing ground truth grid
    
    Returns:
        float: Proportion of correct pixels in the overlapping region
    """
    pred_height = len(prediction)
    pred_width = len(prediction[0]) if pred_height > 0 else 0
    
    gt_height = len(ground_truth)
    gt_width = len(ground_truth[0]) if gt_height > 0 else 0
    
    # If either is empty, return 0
    if pred_height == 0 or pred_width == 0 or gt_height == 0 or gt_width == 0:
        return 0.0
    
    # Find overlapping region
    overlap_height = min(pred_height, gt_height)
    overlap_width = min(pred_width, gt_width)
    
    # Count correct pixels in overlap
    correct = 0
    total = gt_height * gt_width  # Always score against full ground truth size
    
    for i in range(gt_height):
        for j in range(gt_width):
            # Get ground truth value
            gt_value = ground_truth[i][j]
            
            # Get prediction value (0 if out of bounds)
            if i < pred_height and j < pred_width:
                pred_value = prediction[i][j]
            else:
                pred_value = -1  # Out of bounds, will not match
            
            if pred_value == gt_value:
                correct += 1
    
    return correct / total if total > 0 else 0.0


def print_summary(scores):
    """Print a nicely formatted summary of scores."""
    print("\n" + "="*70)
    print("SCORING SUMMARY")
    print("="*70)
    
    print(f"\nTotal Tasks Scored: {scores['total_tasks_scored']}")
    print(f"\nExact Match Score: {scores['total_exact_match_score']:.2f} / {scores['total_tasks_scored']} ({scores['exact_match_percentage']:.2f}%)")
    print(f"Pixel Accuracy: {scores['pixels_correct']} / {scores['total_pixels']} ({scores['pixel_accuracy_percentage']:.2f}%)")
    
    if 'task_scores' in scores:
        print("\n" + "-"*70)
        print("PER-TASK BREAKDOWN:")
        print("-"*70)
        print(f"{'Task ID':<40} {'Exact':<10} {'Pixel Acc':<12} {'Pixels'}")
        print("-"*70)
        
        for task_id, task_score in scores['task_scores'].items():
            exact = "✓" if task_score['exact_match'] == 1.0 else "✗"
            pixel_pct = task_score['pixel_accuracy'] * 100
            pixels_str = f"{task_score['pixels_correct']}/{task_score['total_pixels']}"
            print(f"{task_id:<40} {exact:<10} {pixel_pct:>6.2f}%      {pixels_str}")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score ARC-AGI submission with pixel accuracy')
    parser.add_argument('--submission', default='./submission.json', help='Submission file path')
    parser.add_argument('--solutions', default='dataset/arc-agi_training_solutions.json', help='Solutions file path')
    parser.add_argument('--detailed', action='store_true', help='Show per-task breakdown')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    submission_file_name = args.submission
    solutions_file_name = args.solutions
    
    scores = score_submission_enhanced(
        submission_file_name, 
        solutions_file_name, 
        include_task_scores=args.detailed
    )
    
    if args.json:
        print(json.dumps(scores, indent=2))
    else:
        print_summary(scores)
