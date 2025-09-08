#!/usr/bin/env python3
"""
LLM-Based QA Evaluation System

Uses an LLM as a judge to evaluate question-answering performance,
similar to how Claude Code evaluates internally. This provides more
intelligent evaluation than simple string matching.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class EvaluationResult:
    """Result of LLM evaluation for a single question"""
    question_num: int
    question_id: str
    question_text: str
    expected_answers: List[str]
    agent_response: str
    evaluation: str  # CORRECT, INCORRECT, PARTIAL
    confidence: str  # HIGH, MEDIUM, LOW
    explanation: str
    llm_reasoning: str
    timestamp: str

@dataclass
class OverallResults:
    """Overall evaluation metrics"""
    total_questions: int
    correct: int
    partial: int
    incorrect: int
    accuracy: float  # (correct + 0.5*partial) / total
    strict_accuracy: float  # correct / total
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    evaluation_model: str
    detailed_results: List[EvaluationResult]

class LLMQAEvaluator:
    """LLM-based QA evaluation system"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize evaluator with specified model"""
        self.model = model
        
        # Setup client - check for OpenRouter first, then OpenAI
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if openrouter_key:
            # Use OpenRouter
            self.client = openai.OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
            # Map model names for OpenRouter
            if model == "gpt-4o-mini":
                self.model = "openai/gpt-4o-mini"
            elif model == "gpt-4o":
                self.model = "openai/gpt-4o"
            print(f"Using OpenRouter with model: {self.model}")
        elif openai_key:
            # Use OpenAI directly
            self.client = openai.OpenAI(api_key=openai_key)
            print(f"Using OpenAI with model: {self.model}")
        else:
            raise ValueError("API key required. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")
        
        # Cache for results
        self.cache_file = None
        self.cache = {}
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load QA results from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load results file: {e}")
    
    def setup_cache(self, results_file: str):
        """Setup caching to avoid re-evaluating questions"""
        cache_dir = Path("evaluation_cache")
        cache_dir.mkdir(exist_ok=True)
        
        base_name = Path(results_file).stem
        self.cache_file = cache_dir / f"{base_name}_llm_eval_cache.json"
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached evaluations")
            except:
                self.cache = {}
    
    def save_cache(self):
        """Save cache to disk"""
        if self.cache_file:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def create_evaluation_prompt(self, question: str, expected_answers: List[str], agent_response: str, is_impossible: bool = False) -> str:
        """Create evaluation prompt for LLM judge"""
        
        if is_impossible:
            return f"""You are evaluating a legal document QA system. This question is marked as "impossible" to answer from the document.

QUESTION: {question}

EXPECTED BEHAVIOR: The agent should recognize this question cannot be answered from the document and respond appropriately. Acceptable responses include:
- Explicitly stating "not found", "not specified", "cannot determine from available text"
- Stating "No [X] provisions were found" or "The document does not contain [X]" 
- Concluding that the requested information is absent after thorough analysis
- Any response that acknowledges the absence or unavailability of the requested information

AGENT RESPONSE: {agent_response}

EVALUATION CRITERIA FOR IMPOSSIBLE QUESTIONS:
- CORRECT: Agent recognizes the information is not available/found, whether explicitly ("not found") or through analysis ("no provisions found")
- CORRECT: Agent avoids fabricating specific details that aren't in the document
- INCORRECT: Agent provides specific details, provisions, or quotes that appear to be made up
- INCORRECT: Agent claims definitive information exists when it should not be determinable

IMPORTANT: If the agent says "No [X] found" or "Document does not contain [X]" after searching, this is CORRECT behavior for impossible questions.

Respond in this exact format:
EVALUATION: [CORRECT/INCORRECT]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your detailed explanation of why this evaluation is correct, considering this is an impossible question]"""

        expected_list = "\n".join([f"- {ans}" for ans in expected_answers])
        
        return f"""You are evaluating a legal document QA system's performance. Compare the expected answers with the agent's actual response.

QUESTION: {question}

EXPECTED ANSWERS:
{expected_list}

AGENT RESPONSE: {agent_response}

EVALUATION CRITERIA:
1. Does the agent response contain all the expected information (even if paraphrased)?
2. Are the facts correct and complete?
3. Consider legal terminology variations (e.g., "Company" vs "Electric City Corp.")
4. Consider date format variations (e.g., "September 7, 1999" vs "7th day of September, 1999")
5. The response may contain additional analysis - focus on whether the core expected answers are present

EVALUATION OPTIONS:
- CORRECT: All expected answers are present and accurate
- PARTIAL: Some expected answers are present, but some are missing or unclear
- INCORRECT: Expected answers are missing, wrong, or significantly inaccurate

CONFIDENCE LEVELS:
- HIGH: Very confident in the evaluation
- MEDIUM: Somewhat confident, minor ambiguity
- LOW: Uncertain, significant ambiguity

Respond in this exact format:
EVALUATION: [CORRECT/PARTIAL/INCORRECT]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your detailed explanation of why this evaluation is correct, specifically noting which expected answers were found/missing]"""
    
    def evaluate_single_question(self, question_data: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single question using LLM judge"""
        
        question_id = question_data.get('question_id', '')
        
        # Check cache first
        if question_id in self.cache:
            cached = self.cache[question_id]
            return EvaluationResult(**cached)
        
        question_num = question_data.get('question_num', 0)
        question_text = question_data.get('question_full', '')
        expected_answers = question_data.get('expected_answers', [])
        agent_response = question_data.get('agent_response_full', '')
        is_impossible = question_data.get('is_impossible', False)
        
        # Create evaluation prompt
        prompt = self.create_evaluation_prompt(question_text, expected_answers, agent_response, is_impossible)
        
        # Get LLM evaluation
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of question-answering systems, particularly skilled in legal document analysis. You provide precise, objective evaluations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse LLM response
            evaluation, confidence, reasoning = self._parse_llm_response(llm_response)
            
        except Exception as e:
            print(f"Error evaluating question {question_num}: {e}")
            evaluation = "INCORRECT"
            confidence = "LOW"
            reasoning = f"Evaluation failed due to error: {e}"
            llm_response = f"ERROR: {e}"
        
        result = EvaluationResult(
            question_num=question_num,
            question_id=question_id,
            question_text=question_text,
            expected_answers=expected_answers,
            agent_response=agent_response,
            evaluation=evaluation,
            confidence=confidence,
            explanation=reasoning,
            llm_reasoning=llm_response,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache the result
        self.cache[question_id] = asdict(result)
        
        return result
    
    def _parse_llm_response(self, response: str) -> tuple[str, str, str]:
        """Parse LLM evaluation response"""
        lines = response.strip().split('\n')
        
        evaluation = "INCORRECT"
        confidence = "LOW"
        reasoning = "Failed to parse LLM response"
        
        for line in lines:
            if line.startswith("EVALUATION:"):
                evaluation = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return evaluation, confidence, reasoning
    
    def evaluate_all(self, results_data: Dict[str, Any], save_progress: bool = True) -> OverallResults:
        """Evaluate all questions in the results"""
        
        print(f"Evaluating {len(results_data.get('results', []))} questions using {self.model}")
        
        detailed_results = []
        
        for i, result in enumerate(results_data.get('results', []), 1):
            print(f"Evaluating question {i}/{len(results_data['results'])}: {result.get('question_id', 'Unknown')}")
            
            eval_result = self.evaluate_single_question(result)
            detailed_results.append(eval_result)
            
            # Save progress periodically
            if save_progress and i % 5 == 0:
                self.save_cache()
                print(f"Progress saved. Completed {i} questions.")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Final save
        if save_progress:
            self.save_cache()
        
        return self._calculate_overall_metrics(detailed_results)
    
    def _calculate_overall_metrics(self, detailed_results: List[EvaluationResult]) -> OverallResults:
        """Calculate overall evaluation metrics"""
        
        total = len(detailed_results)
        correct = sum(1 for r in detailed_results if r.evaluation == "CORRECT")
        partial = sum(1 for r in detailed_results if r.evaluation == "PARTIAL")
        incorrect = sum(1 for r in detailed_results if r.evaluation == "INCORRECT")
        
        # Calculate accuracy (partial counts as 0.5)
        accuracy = (correct + 0.5 * partial) / total if total > 0 else 0.0
        strict_accuracy = correct / total if total > 0 else 0.0
        
        # Confidence distribution
        high_conf = sum(1 for r in detailed_results if r.confidence == "HIGH")
        medium_conf = sum(1 for r in detailed_results if r.confidence == "MEDIUM")
        low_conf = sum(1 for r in detailed_results if r.confidence == "LOW")
        
        return OverallResults(
            total_questions=total,
            correct=correct,
            partial=partial,
            incorrect=incorrect,
            accuracy=accuracy,
            strict_accuracy=strict_accuracy,
            high_confidence=high_conf,
            medium_confidence=medium_conf,
            low_confidence=low_conf,
            evaluation_model=self.model,
            detailed_results=detailed_results
        )
    
    def generate_report(self, results: OverallResults, output_file: Optional[str] = None) -> str:
        """Generate detailed evaluation report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("LLM-BASED QA EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Evaluation Model: {results.evaluation_model}")
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append(f"  Total Questions: {results.total_questions}")
        report_lines.append(f"  Correct: {results.correct}")
        report_lines.append(f"  Partial: {results.partial}")
        report_lines.append(f"  Incorrect: {results.incorrect}")
        report_lines.append(f"  Accuracy (with partial): {results.accuracy:.1%}")
        report_lines.append(f"  Strict Accuracy: {results.strict_accuracy:.1%}")
        report_lines.append("")
        
        # Confidence distribution
        report_lines.append("CONFIDENCE DISTRIBUTION:")
        report_lines.append(f"  High Confidence: {results.high_confidence}")
        report_lines.append(f"  Medium Confidence: {results.medium_confidence}")
        report_lines.append(f"  Low Confidence: {results.low_confidence}")
        report_lines.append("")
        
        # Incorrect answers
        incorrect_results = [r for r in results.detailed_results if r.evaluation == "INCORRECT"]
        if incorrect_results:
            report_lines.append("INCORRECT ANSWERS:")
            report_lines.append("-" * 50)
            for result in incorrect_results:
                report_lines.append(f"Q{result.question_num}: {result.question_id}")
                report_lines.append(f"  Expected: {result.expected_answers}")
                report_lines.append(f"  Confidence: {result.confidence}")
                report_lines.append(f"  Reasoning: {result.explanation}")
                report_lines.append("")
        
        # Partial answers
        partial_results = [r for r in results.detailed_results if r.evaluation == "PARTIAL"]
        if partial_results:
            report_lines.append("PARTIAL ANSWERS:")
            report_lines.append("-" * 50)
            for result in partial_results:
                report_lines.append(f"Q{result.question_num}: {result.question_id}")
                report_lines.append(f"  Expected: {result.expected_answers}")
                report_lines.append(f"  Reasoning: {result.explanation}")
                report_lines.append("")
        
        # Low confidence results (potential review needed)
        low_conf_results = [r for r in results.detailed_results if r.confidence == "LOW"]
        if low_conf_results:
            report_lines.append("LOW CONFIDENCE EVALUATIONS (REVIEW RECOMMENDED):")
            report_lines.append("-" * 60)
            for result in low_conf_results:
                report_lines.append(f"Q{result.question_num}: {result.question_id}")
                report_lines.append(f"  Evaluation: {result.evaluation}")
                report_lines.append(f"  Reasoning: {result.explanation}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def save_detailed_results(self, results: OverallResults, output_file: str):
        """Save detailed results to JSON"""
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluation_model": results.evaluation_model,
                "total_questions": results.total_questions,
                "accuracy": results.accuracy,
                "strict_accuracy": results.strict_accuracy
            },
            "summary": {
                "correct": results.correct,
                "partial": results.partial,
                "incorrect": results.incorrect,
                "high_confidence": results.high_confidence,
                "medium_confidence": results.medium_confidence,
                "low_confidence": results.low_confidence
            },
            "detailed_results": [asdict(r) for r in results.detailed_results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """Main evaluation function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_qa_evaluator.py <results_file.json> [model_name]")
        print("Models: gpt-4o-mini (default), gpt-4o, gpt-3.5-turbo")
        sys.exit(1)
    
    results_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o-mini"
    
    evaluator = LLMQAEvaluator(model=model)
    
    try:
        # Setup cache
        evaluator.setup_cache(results_file)
        
        # Load results
        print(f"Loading results from {results_file}...")
        results_data = evaluator.load_results(results_file)
        
        # Run evaluation
        print(f"Running LLM evaluation with {model}...")
        evaluation = evaluator.evaluate_all(results_data)
        
        # Generate reports
        base_name = Path(results_file).stem
        report_file = f"{base_name}_llm_evaluation_report.txt"
        json_file = f"{base_name}_llm_evaluation_detailed.json"
        
        report = evaluator.generate_report(evaluation, report_file)
        evaluator.save_detailed_results(evaluation, json_file)
        
        print(f"\nEvaluation complete!")
        print(f"Model: {model}")
        print(f"Accuracy (with partial): {evaluation.accuracy:.1%}")
        print(f"Strict Accuracy: {evaluation.strict_accuracy:.1%}")
        print(f"Report saved to: {report_file}")
        print(f"Detailed results: {json_file}")
        
        # Print quick summary
        print("\n" + "=" * 60)
        print("QUICK SUMMARY:")
        print(f"Total Questions: {evaluation.total_questions}")
        print(f"Correct: {evaluation.correct}")
        print(f"Partial: {evaluation.partial}")
        print(f"Incorrect: {evaluation.incorrect}")
        print(f"Accuracy: {evaluation.accuracy:.1%}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()