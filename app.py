# app.py
import os
import time
import uuid
import io
import re
import mimetypes
import difflib
import logging
import json
from typing import List, Tuple, Optional
import html as html_escape

import pandas as pd
from flask import (
    Flask, render_template, render_template_string,
    request, redirect, url_for, flash, Response, jsonify
)
from jinja2 import TemplateNotFound
from email.message import EmailMessage
import smtplib

# -----------------------
# Config & logging
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# 🔐 SECRET KEY (FROM ENV)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

APP_NAME = "MailDesk"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SIGNATURE_STORE_FILENAME = os.path.join(app.root_path, "signature_store.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATED_REPORTS = {}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# -----------------------
# Helper functions
# -----------------------
def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(str(email).strip()))

def sanitize_header_value(val: str) -> str:
    return str(val or "").replace("\n", " ").replace("\r", " ").strip()

def guess_mime_type(fname: str):
    ctype, _ = mimetypes.guess_type(fname)
    if ctype:
        return ctype.split("/", 1)
    return "application", "octet-stream"

def _ensure_signature_store():
    if not os.path.exists(SIGNATURE_STORE_FILENAME):
        with open(SIGNATURE_STORE_FILENAME, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_signature_store():
    _ensure_signature_store()
    with open(SIGNATURE_STORE_FILENAME, "r", encoding="utf-8") as f:
        return json.load(f)

def save_signature(email, signature):
    store = load_signature_store()
    store[email.lower()] = signature
    with open(SIGNATURE_STORE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

def get_signature(email):
    return load_signature_store().get(email.lower())

def find_matching_pdf(emp_name, pdfs):
    key = "".join(emp_name.lower().split())
    for fname, data in pdfs:
        base = "".join(os.path.splitext(fname)[0].lower().split())
        if key in base:
            return fname, data
    return None

def build_body(template, emp_name):
    first = emp_name.split()[0] if emp_name else ""
    return template.replace("{name}", first).replace("{full_name}", emp_name)

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        smtp_email = request.form.get("smtp_email")
        smtp_pass = request.form.get("smtp_pass")
        sender_name = request.form.get("sender_name") or APP_NAME
        subject = request.form.get("subject") or "Notification"
        body_template = request.form.get("body") or ""
        signature = request.form.get("signature") or ""

        if request.form.get("save_default_signature") == "on":
            save_signature(smtp_email, signature)

        excel = request.files.get("employees_file")
        pdf_files = request.files.getlist("pdf_files")

        if not excel:
            flash("Employees file required", "error")
            return redirect(url_for("index"))

        df = pd.read_excel(excel) if excel.filename.endswith("xlsx") else pd.read_csv(excel)

        email_col = next((c for c in df.columns if c.lower() in ["email", "mail", "email_id"]), None)
        name_col = next((c for c in df.columns if c.lower() in ["name", "emp_name"]), None)

        pdf_store = [(f.filename, f.read()) for f in pdf_files if f.filename]

        sent, errors = 0, 0
        report = []

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(smtp_email, smtp_pass)

            for _, row in df.iterrows():
                to = row[email_col]
                emp_name = row[name_col] if name_col else ""

                if not is_valid_email(to):
                    errors += 1
                    report.append({"emp_name": emp_name, "email": to, "status": "Invalid Email"})
                    continue

                msg = EmailMessage()
                msg["From"] = f"{sender_name} <{smtp_email}>"
                msg["To"] = to
                msg["Subject"] = sanitize_header_value(subject)

                body = build_body(body_template, emp_name)
                msg.set_content(f"{body}\n\n{signature}")

                matched = find_matching_pdf(emp_name, pdf_store)
                if matched:
                    fname, data = matched
                    m, s = guess_mime_type(fname)
                    msg.add_attachment(data, maintype=m, subtype=s, filename=fname)

                try:
                    server.send_message(msg)
                    sent += 1
                    report.append({"emp_name": emp_name, "email": to, "status": "Sent"})
                except Exception as e:
                    errors += 1
                    report.append({"emp_name": emp_name, "email": to, "status": str(e)})

        flash(f"Completed. Sent: {sent}, Errors: {errors}", "success")
        return render_template("send_email.html", report_rows=report)

    return render_template("send_email.html")


# -----------------------
# Run (Production safe)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
