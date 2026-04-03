import React, { useEffect, useState } from 'react';
import { QUIZ_IMAGES, WASTE_CLASSES } from '../data/quizImages';
import { getDisposalRules } from '../api/client';
import './QuizMode.css';

const QuizMode = ({ onNavigateBack }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState(null);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [disposalRules, setDisposalRules] = useState({});

  // Load disposal rules for feedback
  useEffect(() => {
    const loadRules = async () => {
      try {
        const rules = await getDisposalRules();
        setDisposalRules(rules);
      } catch (error) {
        console.error('Failed to load disposal rules:', error);
      }
    };

    loadRules();
  }, []);

  const currentQuestion = QUIZ_IMAGES[currentQuestionIndex];
  const totalQuestions = QUIZ_IMAGES.length;

  const handleAnswerClick = (selectedClass) => {
    if (showAnswer) return; // Prevent multiple answers

    const isCorrect = selectedClass === currentQuestion.correctClass;
    setUserAnswer(selectedClass);
    setShowAnswer(true);

    if (isCorrect) {
      setScore((prev) => prev + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex + 1 < totalQuestions) {
      setCurrentQuestionIndex((prev) => prev + 1);
      setShowAnswer(false);
      setUserAnswer(null);
    } else {
      setQuizCompleted(true);
    }
  };

  const handleRestartQuiz = () => {
    setCurrentQuestionIndex(0);
    setScore(0);
    setShowAnswer(false);
    setUserAnswer(null);
    setQuizCompleted(false);
  };

  // Get button style based on answer state
  const getButtonClass = (wasteClass) => {
    if (!showAnswer) return 'quiz-answer-button';

    if (wasteClass === currentQuestion.correctClass) {
      return 'quiz-answer-button correct';
    }

    if (wasteClass === userAnswer && wasteClass !== currentQuestion.correctClass) {
      return 'quiz-answer-button incorrect';
    }

    return 'quiz-answer-button disabled';
  };

  // Quiz completion screen
  if (quizCompleted) {
    const percentage = Math.round((score / totalQuestions) * 100);
    let message = '';
    let emoji = '';

    if (percentage >= 90) {
      message = 'Outstanding! You\'re a waste segregation expert!';
      emoji = '🏆';
    } else if (percentage >= 70) {
      message = 'Great job! You have a solid understanding!';
      emoji = '🎉';
    } else if (percentage >= 50) {
      message = 'Good effort! Keep learning to improve!';
      emoji = '👍';
    } else {
      message = 'Keep practicing! Review the guidelines and try again!';
      emoji = '📚';
    }

    return (
      <section className="quiz-mode">
        <div className="quiz-header">
          <h2>📚 Learn Waste Segregation - Quiz Complete!</h2>
          {onNavigateBack && (
            <button onClick={onNavigateBack} className="back-button">
              ← Back to Classifier
            </button>
          )}
        </div>

        <div className="quiz-complete-card">
          <div className="quiz-emoji">{emoji}</div>
          <h3>Quiz Completed!</h3>
          <div className="quiz-final-score">
            <span className="score-number">{score}</span>
            <span className="score-divider">/</span>
            <span className="score-total">{totalQuestions}</span>
          </div>
          <p className="quiz-percentage">{percentage}% Correct</p>
          <p className="quiz-message">{message}</p>

          <div className="quiz-complete-actions">
            <button onClick={handleRestartQuiz} className="restart-button">
              🔄 Restart Quiz
            </button>
            {onNavigateBack && (
              <button onClick={onNavigateBack} className="home-button">
                🏠 Back to Classifier
              </button>
            )}
          </div>
        </div>
      </section>
    );
  }

  // Get disposal info for correct answer
  const correctAnswerInfo = disposalRules[currentQuestion.correctClass] || {
    title: currentQuestion.correctClass,
    description: 'Check municipal guidelines for proper disposal.'
  };

  return (
    <section className="quiz-mode">
      <div className="quiz-header">
        <div>
          <h2>📚 Learn Waste Segregation</h2>
          <p className="quiz-subtitle">Test your knowledge and learn proper disposal methods!</p>
        </div>
        {onNavigateBack && (
          <button onClick={onNavigateBack} className="back-button">
            ← Back to Classifier
          </button>
        )}
      </div>

      <div className="quiz-progress">
        <div className="progress-text">
          Question {currentQuestionIndex + 1} of {totalQuestions}
        </div>
        <div className="progress-bar-container">
          <div
            className="progress-bar"
            style={{ width: `${((currentQuestionIndex + 1) / totalQuestions) * 100}%` }}
          />
        </div>
        <div className="score-display">
          Score: <strong>{score}/{currentQuestionIndex + (showAnswer ? 1 : 0)}</strong>
        </div>
      </div>

      <div className="quiz-content">
        <div className="quiz-image-container">
          <img
            src={currentQuestion.image}
            alt={`Quiz question ${currentQuestionIndex + 1}`}
            className="quiz-image"
          />
          {!showAnswer && currentQuestion.hint && (
            <p className="quiz-hint">💡 Hint: {currentQuestion.hint}</p>
          )}
        </div>

        <div className="quiz-question-section">
          <h3>Which waste category does this item belong to?</h3>

          <div className="quiz-answers-grid">
            {WASTE_CLASSES.map((wasteClass) => (
              <button
                key={wasteClass}
                onClick={() => handleAnswerClick(wasteClass)}
                disabled={showAnswer}
                className={getButtonClass(wasteClass)}
              >
                {wasteClass.replace('-', ' ').toUpperCase()}
              </button>
            ))}
          </div>

          {showAnswer && (
            <div className="quiz-feedback">
              <div className={`feedback-result ${userAnswer === currentQuestion.correctClass ? 'correct' : 'incorrect'}`}>
                {userAnswer === currentQuestion.correctClass ? (
                  <>
                    <span className="feedback-icon">✓</span>
                    <span>Correct!</span>
                  </>
                ) : (
                  <>
                    <span className="feedback-icon">✗</span>
                    <span>Incorrect. The correct answer is: <strong>{currentQuestion.correctClass}</strong></span>
                  </>
                )}
              </div>

              <div className="feedback-disposal-info">
                <h4>{correctAnswerInfo.title}</h4>
                <p>{correctAnswerInfo.description}</p>
              </div>

              <button onClick={handleNextQuestion} className="next-button">
                {currentQuestionIndex + 1 < totalQuestions ? 'Next Question →' : 'See Results →'}
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default QuizMode;
