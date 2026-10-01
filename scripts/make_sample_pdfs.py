"""Generate three small, fictional PDFs used for the demo, tests and evaluation.

Run: python scripts/make_sample_pdfs.py
"""
from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

DOCS = {
    "orion_employee_handbook.pdf": {
        "title": "Orion Labs Employee Handbook",
        "pages": [
            [
                ("Working Hours", "Orion Labs operates on a flexible schedule. Core collaboration hours are 10:00 AM to 3:00 PM Eastern Time, Monday through Thursday. Fridays are reserved for focused work and no internal meetings may be scheduled on Fridays. Employees may work remotely up to three days per week; teams agree on shared office days with their manager."),
                ("Paid Time Off", "Full-time employees accrue 1.75 days of paid time off per month, for a total of 21 days per year. Unused PTO rolls over up to a maximum of 10 days into the following calendar year. Any balance above 10 days is forfeited on January 1. Requests longer than five consecutive working days must be submitted at least three weeks in advance."),
                ("Holidays", "The company observes eleven paid holidays each year, including a floating holiday that each employee may take on a day of personal or cultural significance. The office is closed between December 24 and January 1."),
            ],
            [
                ("Parental Leave", "Orion Labs provides 16 weeks of fully paid parental leave to all parents, including birth, adoptive and foster parents. Leave may be taken in up to two blocks within the first year after the child arrives. Employees become eligible after 90 days of employment."),
                ("Learning Budget", "Each employee receives an annual learning budget of $1,500 for courses, books, certifications and conferences. Unused learning budget does not roll over. Purchases above $300 require manager approval before payment."),
                ("Equipment", "New hires choose either a 14-inch or 16-inch laptop and receive a $400 stipend for home office furniture. Equipment remains company property and must be returned within ten business days of an employee's last day."),
            ],
            [
                ("Expense Reimbursement", "Business expenses must be submitted through the Ledger app within 30 days of purchase with an itemized receipt attached. Meals while traveling are reimbursed up to $75 per day. Economy class is required for flights shorter than six hours; premium economy is allowed for longer flights."),
                ("Performance Reviews", "Performance reviews happen twice a year, in March and September. Each review includes a self-assessment, peer feedback from at least two colleagues and a calibration session among managers. Compensation adjustments are announced in April."),
            ],
        ],
    },
    "nimbus_storage_manual.pdf": {
        "title": "Nimbus Cloud Storage - Product Manual",
        "pages": [
            [
                ("Overview", "Nimbus is an object storage service for teams. Files are stored in buckets, and every bucket belongs to exactly one project. Each object can be up to 5 TB in size. Objects larger than 100 MB should be uploaded with multipart upload, which splits the file into parts of 8 MB to 512 MB that are uploaded in parallel."),
                ("Storage Classes", "Nimbus offers three storage classes. Hot storage costs $0.023 per GB per month and is designed for frequently accessed data. Cool storage costs $0.010 per GB per month and has a minimum storage duration of 30 days. Archive storage costs $0.002 per GB per month, has a minimum duration of 180 days, and retrieval takes up to 12 hours."),
            ],
            [
                ("Versioning", "When versioning is enabled on a bucket, Nimbus keeps every version of an object. Deleting an object adds a delete marker instead of removing data, so previous versions can be restored. Lifecycle rules can automatically delete non-current versions after a chosen number of days."),
                ("Encryption", "All objects are encrypted at rest with AES-256 by default. Customers on the Business plan can supply their own keys through the Nimbus Key Vault. Data in transit is protected with TLS 1.3, and connections using TLS versions older than 1.2 are rejected."),
                ("Rate Limits", "Each bucket supports up to 3,500 write requests and 5,500 read requests per second per prefix. Clients that exceed the limit receive HTTP 429 responses and should retry with exponential backoff starting at 100 milliseconds."),
            ],
            [
                ("Pricing Plans", "The Starter plan is free and includes 10 GB of hot storage and 1 GB of daily egress. The Team plan costs $20 per user per month and includes 1 TB of pooled storage. The Business plan adds customer-managed keys, single sign-on and a 99.99 percent availability SLA."),
                ("Support", "Starter customers receive community forum support. Team customers get email support with a response time of one business day. Business customers have 24/7 phone support with a one-hour response time for critical incidents."),
            ],
        ],
    },
    "helix_security_policy.pdf": {
        "title": "Helix Corp Information Security Policy",
        "pages": [
            [
                ("Passwords", "Passwords for company accounts must be at least 14 characters long. Passwords do not expire on a schedule but must be changed immediately if a compromise is suspected. Employees must use the approved password manager, and reusing a company password on any other service is prohibited."),
                ("Multi-Factor Authentication", "Multi-factor authentication is mandatory for email, source control, cloud consoles and the VPN. Hardware security keys are required for administrators and for anyone with production database access. SMS codes are not an accepted second factor."),
            ],
            [
                ("Data Classification", "Helix classifies data into four levels: Public, Internal, Confidential and Restricted. Restricted data includes customer payment information and health records and may only be stored in systems approved by the security team. Confidential data must never be pasted into external AI tools unless the tool has been approved by the security team."),
                ("Incident Reporting", "Suspected security incidents must be reported to the security team within one hour of discovery by emailing the security inbox or calling the on-call number. Employees should not attempt to investigate or clean up a compromised device themselves; they should disconnect it from the network and wait for instructions."),
            ],
            [
                ("Device Security", "Company laptops must have full-disk encryption enabled and lock automatically after five minutes of inactivity. Operating system updates must be installed within 14 days of release, and critical security patches within 72 hours. Lost or stolen devices must be reported within 24 hours."),
                ("Access Reviews", "Managers review their team's access to systems every quarter. Access for departing employees is revoked on their last working day, and shared accounts are not permitted except for documented service accounts."),
            ],
        ],
    },
}


def build(name: str, spec: dict) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, sections in enumerate(spec["pages"]):
        pdf.add_page()
        if i == 0:
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 10, spec["title"], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        for heading, body in sections:
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
    pdf.output(str(OUT_DIR / name))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, spec in DOCS.items():
        build(name, spec)
        print("wrote", OUT_DIR / name)
