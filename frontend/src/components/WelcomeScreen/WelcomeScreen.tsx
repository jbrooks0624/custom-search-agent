import { STARTER_QUESTIONS } from '../../constants'
import './WelcomeScreen.css'

interface WelcomeScreenProps {
  onSelectQuestion: (question: string, isDeepResearch: boolean) => void
}

export function WelcomeScreen({ onSelectQuestion }: WelcomeScreenProps) {
  return (
    <div className="welcome">
      <h1>Custom Search Agent</h1>
      <p className="subtitle">
        Enable deep research mode for comprehensive, well-researched answers
      </p>

      <div className="starter-questions">
        <div className="starter-section">
          <h3>Try Search</h3>
          <div className="starter-list">
            {STARTER_QUESTIONS.standard.map((question, i) => (
              <button
                key={i}
                className="starter-button"
                onClick={() => onSelectQuestion(question, false)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <div className="starter-section">
          <h3>Try Deep Research</h3>
          <div className="starter-list">
            {STARTER_QUESTIONS.deepResearch.map((question, i) => (
              <button
                key={i}
                className="starter-button deep"
                onClick={() => onSelectQuestion(question, true)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
