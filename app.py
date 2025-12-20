# app.py (DEPLOYMENT READY)

import os
import time
import uuid
import io
import re
import mimetypes
import difflib
import logging
from typing import List, Tuple

import pandas as pd
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, Response, jsonify
)
from email.message import EmailMessage
import smtplib

# -----------------------
# App Configuration
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "unsafe-dev-key")

APP_NAME = "MailDesk"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASS = os.environ.get("SMTP_PASS")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

GENERATED_REPORTS = {}

# -----------------------
# Helpers
# -----------------------
def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(str(email).strip()))

def guess_mime_type(fname: str):
    ctype, _ = mimetypes.guess_type(fname)
    if ctype:
        return ctype.split("/", 1)
    return "application", "octet-stream"

def find_matching_pdf(emp_name: str, pdf_store):
    if not emp_name:
        return None
    key = "".join(emp_name.lower().split())
    for fname, data in pdf_store:
        base = "".join(os.path.splitext(fname)[0].lower().split())
        if key in base:
            return fname, data
    return None

def build_body(template: str, emp_name: str):
    first = emp_name.split()[0] if emp_name else ""
    try:
        return template.format(name=first, full_name=emp_name)
    except Exception:
        return template.replace("{name}", first).replace("{full_name}", emp_name)

# -----------------------
# Email Sender
# -----------------------
def send_email(
    server,
    to_email,
    emp_name,
    subject,
    body_template,
    sender_name,
    attachments,
    cc_list,
    bcc_list,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{SMTP_EMAIL}>"
    msg["To"] = to_email
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)

    body = build_body(body_template, emp_name)
    msg.set_content(body)

    for fname, data in attachments:
        maintype, subtype = guess_mime_type(fname)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=fname)

    recipients = [to_email] + cc_list + bcc_list
    server.send_message(msg, from_addr=SMTP_EMAIL, to_addrs=recipients)

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if not SMTP_EMAIL or not SMTP_PASS:
            flash("SMTP credentials not configured on server", "error")
            return redirect(url_for("index"))

        excel_file = request.files.get("employees_file")
        if not excel_file:
            flash("Upload employee Excel/CSV", "error")
            return redirect(url_for("index"))

        subject = request.form.get("subject", "Notification")
        body_template = request.form.get("body", "Hello {name}")
        sender_name = request.form.get("sender_name", APP_NAME)

        cc_list = [x.strip() for x in request.form.get("cc", "").split(",") if x.strip()]
        bcc_list = [x.strip() for x in request.form.get("bcc", "").split(",") if x.strip()]

        # Read employee file
        try:
            if excel_file.filename.lower().endswith(".csv"):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file)
        except Exception as e:
            flash(f"File read error: {e}", "error")
            return redirect(url_for("index"))

        email_col = next((c for c in df.columns if c.lower() in ("email", "email_id", "mail")), None)
        name_col = next((c for c in df.columns if c.lower() in ("name", "emp_name")), None)

        if not email_col:
            flash("Employee file must have email column", "error")
            return redirect(url_for("index"))

        # Read PDFs
        pdf_store = []
        for f in request.files.getlist("pdf_files"):
            if f and f.filename:
                pdf_store.append((f.filename, f.read()))

        sent, errors = 0, 0
        report = []

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)

                for _, row in df.iterrows():
                    to_email = str(row[email_col]).strip()
                    emp_name = str(row[name_col]) if name_col else ""

                    if not is_valid_email(to_email):
                        errors += 1
                        report.append({"email": to_email, "status": "Invalid"})
                        continue

                    attachments = []
                    if pdf_store and name_col:
                        match = find_matching_pdf(emp_name, pdf_store)
                        if match:
                            attachments.append(match)

                    try:
                        send_email(
                            server,
                            to_email,
                            emp_name,
                            subject,
                            body_template,
                            sender_name,
                            attachments,
                            cc_list,
                            bcc_list,
                        )
                        sent += 1
                        report.append({"email": to_email, "status": "Sent"})
                    except Exception as e:
                        errors += 1
                        report.append({"email": to_email, "status": str(e)})

        except smtplib.SMTPAuthenticationError:
            flash("SMTP Authentication Failed", "error")
            return redirect(url_for("index"))

        report_id = str(uuid.uuid4())
        GENERATED_REPORTS[report_id] = report

        flash(f"Sent: {sent}, Errors: {errors}", "success")
        return render_template("send_email.html", report=report, report_id=report_id)

    return render_template("send_email.html")

# -----------------------
# Report Download
# -----------------------
@app.route("/download/<rid>")
def download(rid):
    rows = GENERATED_REPORTS.get(rid)
    if not rows:
        flash("Report expired", "error")
        return redirect(url_for("index"))

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return Response(
        output.read(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=maildesk_report.xlsx"},
    )

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
