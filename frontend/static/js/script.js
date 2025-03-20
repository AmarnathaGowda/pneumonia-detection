document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('prediction').textContent = 'Error: ' + data.error;
            document.getElementById('confidence').textContent = '';
        } else {
            document.getElementById('prediction').textContent = 'Prediction: ' + data.prediction;
            document.getElementById('confidence').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(2) + '%';
        }
    })
    .catch(error => {
        document.getElementById('prediction').textContent = 'Error: ' + error.message;
        document.getElementById('confidence').textContent = '';
    });
});