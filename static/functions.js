function interrupt() {
    fetch("/cancel_task",{method: "POST",})
}


function show_interrupt_button(){
    if (document.getElementById('interrupt').getAttribute('disabled')=== 'disabled'){
        document.getElementById('interrupt').removeAttribute('disabled');
    }
    else {
        document.getElementById('interrupt').setAttribute('disabled','disabled')
    }
}


function run_scopus_search(event){
    event.preventDefault(); 
    var a = document.getElementById("abstract").value; 
    fetch("/scopus_search", {
        method: "POST",
        body: new URLSearchParams({ 'a': a }),
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById("result");

        // Clear any previous content in the div
        resultDiv.innerHTML = '';

        // Create a new <pre> element to format and display the JSON
        var preElement = document.createElement('pre');
        preElement.textContent = JSON.stringify(data, null, 2); // Format the JSON with indentation

        // Append the <pre> element to the 'result' div
        resultDiv.appendChild(preElement);
    })
    .catch(error => console.error(error));
}


function run_patent_search(event){
    event.preventDefault(); // Prevent the default form submission
    var a = document.getElementById("abstract").value; // Get the value of input field 'a'
    // Send 'a' to the Flask backend using a POST request
    fetch("/patent_search", {
        method: "POST",
        body: new URLSearchParams({ 'a': a }),
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById("result");

        // Clear any previous content in the div
        resultDiv.innerHTML = '';

        // Create a new <pre> element to format and display the JSON
        var preElement = document.createElement('pre');
        preElement.textContent = JSON.stringify(data, null, 2); // Format the JSON with indentation

        // Append the <pre> element to the 'result' div
        resultDiv.appendChild(preElement);
    })
    .catch(error => console.error(error));
}