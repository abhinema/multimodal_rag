// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');
const queryInput = document.getElementById('query-input');
const queryBtn = document.getElementById('query-btn');
const resultsContainer = document.getElementById('results-container');
const loading = document.getElementById('loading');

// API endpoint configuration
const API_URL = 'http://localhost:8000'; // Update this with your actual API URL

// File list storage
let files = [];

// Drag and drop functionality
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('active');
}

function unhighlight() {
    dropArea.classList.remove('active');
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const newFiles = [...dt.files];
    handleFiles(newFiles);
}

// Handle file input change
fileInput.addEventListener('change', function() {
    handleFiles([...this.files]);
});

// Process the files
function handleFiles(newFiles) {
    const pdfFiles = newFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length === 0) {
        showMessage('Please upload PDF files only.', 'error');
        return;
    }
    
    // Add files to our list
    pdfFiles.forEach(file => {
        if (!files.some(f => f.name === file.name)) {
            files.push(file);
            displayFile(file);
        }
    });
    
    updateUploadButton();
}

// Display file in the list
function displayFile(file) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const fileName = document.createElement('div');
    fileName.className = 'file-name';
    fileName.textContent = file.name;
    
    const removeBtn = document.createElement('div');
    removeBtn.className = 'remove-file';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', () => {
        files = files.filter(f => f.name !== file.name);
        fileItem.remove();
        updateUploadButton();
    });
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(removeBtn);
    fileList.appendChild(fileItem);
}

// Update upload button state
function updateUploadButton() {
    uploadBtn.disabled = files.length === 0;
}

// Upload and process documents
uploadBtn.addEventListener('click', async () => {
    if (files.length === 0) return;
    
    showLoading();
    
    try {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        
        const result = await response.json();
        files = [];
        fileList.innerHTML = '';
        updateUploadButton();
        
        showMessage('Documents processed successfully!', 'success');
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Failed to process documents: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
});

// Submit a query
queryBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) {
        showMessage('Please enter a question.', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Failed to get results: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
});

// Display results
function displayResults(data) {
    resultsContainer.innerHTML = '';
    
    if (!data || (Array.isArray(data.results) && data.results.length === 0)) {
        resultsContainer.innerHTML = '<p>No results found.</p>';
        return;
    }
    
    // Handle different response formats
    if (data.answer) {
        // If there's a direct answer
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const resultText = document.createElement('div');
        resultText.className = 'result-text';
        resultText.innerHTML = `<strong>Answer:</strong> ${data.answer}`;
        
        resultItem.appendChild(resultText);
        
        if (data.sources && data.sources.length > 0) {
            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';
            sourcesList.innerHTML = '<strong>Sources:</strong>';
            
            const sourceItems = document.createElement('ul');
            data.sources.forEach(source => {
                const sourceItem = document.createElement('li');
                sourceItem.textContent = source;
                sourceItems.appendChild(sourceItem);
            });
            
            sourcesList.appendChild(sourceItems);
            resultItem.appendChild(sourcesList);
        }
        
        resultsContainer.appendChild(resultItem);
    } else if (data.results) {
        // If there's a list of results
        data.results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            const resultText = document.createElement('div');
            resultText.className = 'result-text';
            resultText.textContent = result.content || result.text;
            
            const resultMeta = document.createElement('div');
            resultMeta.className = 'result-meta';
            
            if (result.source) {
                resultMeta.textContent = `Source: ${result.source}`;
            }
            
            resultItem.appendChild(resultText);
            resultItem.appendChild(resultMeta);
            resultsContainer.appendChild(resultItem);
        });
    } else {
        // Fallback for other formats
        resultsContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
}

// Show loading spinner
function showLoading() {
    loading.classList.remove('hidden');
}

// Hide loading spinner
function hideLoading() {
    loading.classList.add('hidden');
}

// Show message
function showMessage(message, type) {
    const messageElement = document.createElement('div');
    messageElement.className = `${type}-message`;
    messageElement.textContent = message;
    
    // Remove any existing message
    const existingMessage = document.querySelector(`.${type}-message`);
    if (existingMessage) {
        existingMessage.remove();
    }
    
    if (type === 'error') {
        resultsContainer.prepend(messageElement);
    } else {
        resultsContainer.prepend(messageElement);
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        messageElement.remove();
    }, 5000);
}