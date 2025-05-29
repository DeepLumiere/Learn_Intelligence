# Security Policy

## Supported Versions

This repository contains various machine learning models. Due to the nature of these models often relying on specific library versions and frameworks, direct "version support" like traditional software isn't directly applicable in the same way.

However, we endeavor to ensure the **latest stable versions of the primary dependencies** used by our models are considered "supported." This means we aim to keep the models compatible with recent releases of frameworks like TensorFlow, PyTorch, scikit-learn, and other core libraries.

**Generally, we support the following:**

* **Models trained with the latest stable versions of their respective ML frameworks.**
* **Models actively maintained and updated within the last 12 months.**

**Unsupported versions include:**

* Models or dependencies that are explicitly deprecated by their upstream projects.
* Models that have not been updated or tested with recent framework versions for an extended period (typically over 12-18 months).

**Specific version details will be provided within each model's respective directory or README file.**

## Reporting a Vulnerability

We take the security of our machine learning models and the data they process seriously. While the primary security concerns for ML models often revolve around data poisoning, adversarial attacks, and privacy breaches, traditional software vulnerabilities in the underlying code or dependencies are also critical.

If you believe you have found a security vulnerability in any of the models, scripts, or dependencies within this repository, please report it to us as soon as possible.

**How to Report a Vulnerability:**

1.  **Do NOT open a public GitHub issue.** Please report vulnerabilities privately to prevent malicious actors from exploiting them.
2.  **Email us:** Send an email to `deepensify@gmail.com`.
3.  **Include the following information in your report:**
    * **Description of the vulnerability:** Explain the nature of the vulnerability, its potential impact, and how it can be exploited.
    * **Affected model(s) or file(s):** Specify which model(s) or files in the repository are affected.
    * **Steps to reproduce:** Provide clear and concise steps to reproduce the vulnerability. This is crucial for us to verify and fix the issue.
    * **Proof of concept (if applicable):** If you have a proof-of-concept code or example, please include it.
    * **Your contact information:** So we can follow up with you.

**Our Response Process:**

* **Acknowledgement:** You can expect an initial acknowledgement of your report within **3 business days**.
* **Investigation:** We will investigate the reported vulnerability thoroughly. This may involve collaborating with you for further details.
* **Status Updates:** We will provide you with updates on the progress of our investigation and remediation efforts, typically every **7-14 business days**, depending on the complexity of the issue.
* **Disclosure:** Once the vulnerability is patched, we will coordinate with you regarding public disclosure, if appropriate. We believe in responsible disclosure to ensure the security of all users.
* **Recognition:** We appreciate the efforts of security researchers and may offer public recognition for valuable contributions, with your consent.

**What to Expect if the Vulnerability is Accepted or Declined:**

* **Accepted:** If the vulnerability is accepted, we will prioritize its fix and work to release an update as soon as possible. We will keep you informed of our progress.
* **Declined:** If the vulnerability is declined (e.g., it's not a security vulnerability, or it's a known issue with a documented workaround), we will provide a clear explanation for our decision.

Thank you for helping to keep our machine learning models secure.
