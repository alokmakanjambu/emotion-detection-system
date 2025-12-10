/**
 * Emotion Detection Frontend Application
 * Connects to FastAPI backend for text emotion analysis
 */

// ===== Configuration =====
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:8000/api/v1',
    ENDPOINTS: {
        id: '/predict/id',
        en: '/predict',
        health: '/health',
        healthId: '/health/id'
    }
};

// ===== Emotion Data =====
const EMOTIONS = {
    joy: { emoji: '😊', color: '#fbbf24', label: { id: 'Senang', en: 'Joy' } },
    sadness: { emoji: '😢', color: '#60a5fa', label: { id: 'Sedih', en: 'Sadness' } },
    anger: { emoji: '😠', color: '#f87171', label: { id: 'Marah', en: 'Anger' } },
    fear: { emoji: '😨', color: '#a78bfa', label: { id: 'Takut', en: 'Fear' } },
    love: { emoji: '😍', color: '#fb7185', label: { id: 'Cinta', en: 'Love' } },
    neutral: { emoji: '😐', color: '#94a3b8', label: { id: 'Netral', en: 'Neutral' } },
    surprise: { emoji: '😲', color: '#34d399', label: { id: 'Terkejut', en: 'Surprise' } }
};

// ===== State =====
let state = {
    language: 'id',
    isLoading: false,
    history: JSON.parse(localStorage.getItem('emotionHistory') || '[]')
};

// ===== DOM Elements =====
const elements = {
    textInput: document.getElementById('textInput'),
    charCount: document.getElementById('charCount'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resultSection: document.getElementById('resultSection'),
    emotionEmoji: document.getElementById('emotionEmoji'),
    emotionLabel: document.getElementById('emotionLabel'),
    confidenceBar: document.getElementById('confidenceBar'),
    confidenceText: document.getElementById('confidenceText'),
    probabilitiesChart: document.getElementById('probabilitiesChart'),
    historySection: document.getElementById('historySection'),
    historyList: document.getElementById('historyList'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    apiStatus: document.getElementById('apiStatus'),
    apiStatusText: document.getElementById('apiStatusText'),
    toast: document.getElementById('toast'),
    langBtns: document.querySelectorAll('.lang-btn')
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkApiStatus();
    updateHistoryDisplay();
    updatePlaceholder();
});

function initEventListeners() {
    // Text input
    elements.textInput.addEventListener('input', handleInput);
    elements.textInput.addEventListener('keydown', handleKeyDown);

    // Analyze button
    elements.analyzeBtn.addEventListener('click', analyzeText);

    // Language buttons
    elements.langBtns.forEach(btn => {
        btn.addEventListener('click', () => switchLanguage(btn.dataset.lang));
    });

    // Clear history
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
}

// ===== Input Handlers =====
function handleInput(e) {
    const length = e.target.value.length;
    elements.charCount.textContent = length;

    // Visual feedback for char limit
    if (length > 900) {
        elements.charCount.style.color = '#f59e0b';
    } else if (length > 950) {
        elements.charCount.style.color = '#ef4444';
    } else {
        elements.charCount.style.color = '';
    }
}

function handleKeyDown(e) {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeText();
    }
}

// ===== Language Switch =====
function switchLanguage(lang) {
    state.language = lang;

    // Update button states
    elements.langBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });

    updatePlaceholder();
}

function updatePlaceholder() {
    const placeholders = {
        id: 'Ketik atau paste teks di sini untuk dianalisis emosinya...',
        en: 'Type or paste text here to analyze its emotion...'
    };
    elements.textInput.placeholder = placeholders[state.language];
}

// ===== API Functions =====
async function checkApiStatus() {
    try {
        const [enResponse, idResponse] = await Promise.all([
            fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.health}`).catch(() => null),
            fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.healthId}`).catch(() => null)
        ]);

        const enOk = enResponse?.ok;
        const idOk = idResponse?.ok;

        if (enOk || idOk) {
            elements.apiStatus.className = 'status-dot connected';
            elements.apiStatusText.textContent = 'API Connected';
        } else {
            throw new Error('API not available');
        }
    } catch (error) {
        elements.apiStatus.className = 'status-dot error';
        elements.apiStatusText.textContent = 'API Offline';
    }
}

async function analyzeText() {
    const text = elements.textInput.value.trim();

    if (!text) {
        showToast('Silakan masukkan teks terlebih dahulu', 'error');
        elements.textInput.focus();
        return;
    }

    if (state.isLoading) return;

    setLoading(true);

    try {
        const endpoint = state.language === 'id'
            ? CONFIG.ENDPOINTS.id
            : CONFIG.ENDPOINTS.en;

        const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResult(data);
        addToHistory(data);

    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Gagal menganalisis teks. Pastikan API berjalan.', 'error');
    } finally {
        setLoading(false);
    }
}

// ===== UI Functions =====
function setLoading(loading) {
    state.isLoading = loading;
    elements.analyzeBtn.disabled = loading;
    elements.analyzeBtn.classList.toggle('loading', loading);
}

function displayResult(data) {
    const emotion = data.emotion.toLowerCase();
    const emotionData = EMOTIONS[emotion] || EMOTIONS.neutral;
    const confidence = Math.round(data.confidence * 100);

    // Show result section
    elements.resultSection.classList.remove('hidden');

    // Update emotion display
    elements.emotionEmoji.textContent = emotionData.emoji;
    elements.emotionLabel.textContent = emotionData.label[state.language] || emotion;
    elements.emotionLabel.style.color = emotionData.color;

    // Update confidence bar
    elements.confidenceBar.style.width = `${confidence}%`;
    elements.confidenceBar.style.backgroundColor = emotionData.color;
    elements.confidenceText.textContent = `Confidence: ${confidence}%`;

    // Update probabilities chart
    displayProbabilities(data.probabilities);

    // Scroll to result
    elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayProbabilities(probabilities) {
    elements.probabilitiesChart.innerHTML = '';

    // Sort by probability descending
    const sorted = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a);

    sorted.forEach(([emotion, prob]) => {
        const emotionData = EMOTIONS[emotion] || EMOTIONS.neutral;
        const percentage = Math.round(prob * 100);

        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-label">
                <span>${emotionData.emoji}</span>
                <span>${emotionData.label[state.language] || emotion}</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${percentage}%; background-color: ${emotionData.color}">
                    ${percentage > 10 ? `<span class="prob-value">${percentage}%</span>` : ''}
                </div>
            </div>
        `;

        elements.probabilitiesChart.appendChild(item);
    });
}

// ===== History Functions =====
function addToHistory(data) {
    const historyItem = {
        text: data.text,
        emotion: data.emotion,
        confidence: data.confidence,
        language: state.language,
        timestamp: Date.now()
    };

    // Add to beginning
    state.history.unshift(historyItem);

    // Keep only last 10
    if (state.history.length > 10) {
        state.history.pop();
    }

    // Save to localStorage
    localStorage.setItem('emotionHistory', JSON.stringify(state.history));

    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    if (state.history.length === 0) {
        elements.historySection.classList.add('hidden');
        return;
    }

    elements.historySection.classList.remove('hidden');
    elements.historyList.innerHTML = '';

    state.history.forEach((item, index) => {
        const emotionData = EMOTIONS[item.emotion.toLowerCase()] || EMOTIONS.neutral;
        const truncatedText = item.text.length > 50
            ? item.text.substring(0, 50) + '...'
            : item.text;

        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <span class="history-emoji">${emotionData.emoji}</span>
            <span class="history-text">${truncatedText}</span>
            <span class="history-emotion" style="background-color: ${emotionData.color}20; color: ${emotionData.color}">
                ${item.emotion}
            </span>
        `;

        div.addEventListener('click', () => {
            elements.textInput.value = item.text;
            elements.charCount.textContent = item.text.length;
            elements.textInput.focus();
        });

        elements.historyList.appendChild(div);
    });
}

function clearHistory() {
    state.history = [];
    localStorage.removeItem('emotionHistory');
    updateHistoryDisplay();
    showToast('Riwayat berhasil dihapus', 'success');
}

// ===== Toast Notification =====
function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;

    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// ===== Utility Functions =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
