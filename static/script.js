const form = document.getElementById("loanForm");

form.addEventListener("submit", async function(event) {
    event.preventDefault();

    const button = document.getElementById("predictBtn");
    const result = document.getElementById("result");
    const error = document.getElementById("error");

    button.disabled = true;
    button.textContent = "Predicting...";
    error.textContent = "";

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    data.ApplicantIncome = Number(data.ApplicantIncome);
    data.CoapplicantIncome = Number(data.CoapplicantIncome);
    data.LoanAmount = Number(data.LoanAmount);
    data.Loan_Amount_Term = Number(data.Loan_Amount_Term);
    data.Credit_History = Number(data.Credit_History);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.error || "Prediction failed");
        }

        document.getElementById("prediction").textContent =
            output.loan_prediction;

        document.getElementById("probability").textContent =
            output.approval_probability + "%";

        document.getElementById("riskScore").textContent =
            output.risk_score;

        document.getElementById("riskLevel").textContent =
            output.risk_level;

        document.getElementById("prediction").style.color =
            output.loan_prediction === "Approved" ? "#168451" : "#d93636";

        document.getElementById("riskLevel").style.color =
            output.risk_level === "Low" ? "#168451" :
            output.risk_level === "Medium" ? "#d58b16" : "#d93636";

        result.classList.remove("hidden");
        result.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
        error.textContent = err.message;
    } finally {
        button.disabled = false;
        button.textContent = "Predict Loan Approval →";
    }
});