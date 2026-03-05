document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictBtn");

    form.addEventListener("submit", function () {
        button.innerText = "Analyzing...";
        button.disabled = true;
    });

});