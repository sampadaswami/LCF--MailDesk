"""
Created on Sat Dec 20 19:54:10 2025 

@author: Sampada
"""

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
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    flash,
    Response,
    jsonify,
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
app.secret_key = os.environ.get("FLASK_SECRET", "replace-me-in-dev")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

APP_NAME = "MailDesk"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SIGNATURE_STORE_FILENAME = os.path.join(app.root_path, "signature_store.json")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

GENERATED_REPORTS = {}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# -----------------------
# Template safe helper
# -----------------------
def render_template_safe(template_name: str, **context):
    try:
        context.setdefault("APP_NAME", APP_NAME)
        return render_template(template_name, **context)
    except TemplateNotFound as e:
        expected_path = os.path.join(app.template_folder or "templates", template_name)
        logger.exception("TemplateNotFound: %s (expected at %s)", template_name, expected_path)
        msg = f"""
        <h2>Template not found: {template_name}</h2>
        <p>Expected path: <code>{expected_path}</code></p>
        <p>Make sure your template file exists in the <code>templates</code> folder and is named exactly.</p>
        <hr>
        <p>Context keys: {list(context.keys())}</p>
        """
        return render_template_string(msg), 500


# -----------------------
# Signature store helpers
# -----------------------
def _ensure_signature_store_exists():
    if not os.path.exists(SIGNATURE_STORE_FILENAME):
        try:
            with open(SIGNATURE_STORE_FILENAME, "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            logger.debug("Created signature store at %s", SIGNATURE_STORE_FILENAME)
        except Exception as e:
            logger.exception("Could not create signature store file: %s", e)


def load_signature_store() -> dict:
    _ensure_signature_store_exists()
    try:
        with open(SIGNATURE_STORE_FILENAME, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception as e:
        logger.exception("Could not load signature store: %s", e)
        return {}


def save_signature_for_email(smtp_email: str, signature: str):
    smtp_email = (smtp_email or "").strip().lower()
    if not smtp_email:
        return
    store = load_signature_store()
    store[smtp_email] = signature or ""
    try:
        with open(SIGNATURE_STORE_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2)
        logger.debug("Saved signature for %s", smtp_email)
    except Exception as e:
        logger.exception("Failed to save signature: %s", e)
        raise


def get_saved_signature(smtp_email: str) -> Optional[str]:
    smtp_email = (smtp_email or "").strip().lower()
    if not smtp_email:
        return None
    store = load_signature_store()
    return store.get(smtp_email)


# -----------------------
# Helpers
# -----------------------
def is_valid_email(email: str) -> bool:
    try:
        return bool(EMAIL_RE.match(str(email).strip()))
    except Exception:
        return False


def sanitize_header_value(val: str) -> str:
    if val is None:
        return ""
    return str(val).replace("\n", " ").replace("\r", " ").strip()


def guess_mime_type(fname: str) -> Tuple[str, str]:
    ctype, encoding = mimetypes.guess_type(fname)
    if ctype:
        parts = ctype.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else "octet-stream"
    return "application", "octet-stream"


def find_matching_pdf(emp_name: str, pdf_store: List[Tuple[str, bytes]]):
    if not emp_name:
        return None
    key = "".join(str(emp_name).lower().split())
    # substring match
    for fname, fdata in pdf_store:
        base = os.path.splitext(fname)[0]
        base_norm = "".join(base.lower().split())
        if key in base_norm:
            return fname, fdata
    # fuzzy match
    base_names = [os.path.splitext(fname)[0] for fname, _ in pdf_store]
    matches = difflib.get_close_matches(str(emp_name), base_names, n=1, cutoff=0.7)
    if matches:
        match_base = matches[0]
        for fname, fdata in pdf_store:
            if os.path.splitext(fname)[0] == match_base:
                return fname, fdata
    return None


def purge_old_reports(ttl_seconds: int = 24 * 3600):
    now = time.time()
    for rid in list(GENERATED_REPORTS.keys()):
        entry = GENERATED_REPORTS.get(rid, {})
        ts = entry.get("ts", 0)
        if now - ts > ttl_seconds:
            del GENERATED_REPORTS[rid]


def build_personal_body(
    body_template: str,
    signature: str,
    smtp_email: str,
    sender_name: str,
    emp_name: str,
):
    """
    Build the plain text email body.

    - {name} -> first name only (first token from emp_name)
    - {full_name} -> full emp_name as-is
    """
    body_template = (body_template or "").strip()
    if not body_template:
        body_template = "Hi {name},\n\nPlease see the details below."

    # Prepare name variants
    full_name = str(emp_name or "").strip()
    if full_name:
        # split on whitespace and pick first token as first name
        first_name = full_name.split()[0]
    else:
        first_name = ""

    # Try to format using placeholders. We provide both name and full_name keys.
    try:
        body = body_template.format(name=first_name, full_name=full_name)
    except Exception:
        # If formatting fails (bad placeholders), fall back to simple replacement attempt
        try:
            body = body_template.replace("{name}", first_name).replace("{full_name}", full_name)
        except Exception:
            body = body_template

    signature = (signature or "").strip()
    if not signature:
        signature = f"Regards,\n{sender_name or smtp_email}"
    full_body = f"{body}\n\n{signature}"
    return full_body


def _simple_markdown_to_html(plain_text: str) -> str:
    if plain_text is None:
        return ""
    safe = html_escape.escape(plain_text)
    bold_re = re.compile(r'\*\*(.+?)\*\*', flags=re.DOTALL)
    safe = bold_re.sub(r'<strong>\1</strong>', safe)
    safe = safe.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = [p.strip() for p in safe.split('\n\n') if p.strip() != ""]
    html_parts = []
    for p in paragraphs:
        p = p.replace('\n', '<br/>')
        html_parts.append(f"<p>{p}</p>")
    return "\n".join(html_parts)


def _sanitize_signature_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    # remove script tags
    cleaned = re.sub(r'(?is)<script.*?>.*?</script>', '', raw_html)
    # remove on* event attributes like onclick="..."
    cleaned = re.sub(r'(?i)\s+on\w+\s*=\s*"(?:[^"]*)"', '', cleaned)
    cleaned = re.sub(r"(?i)\s+on\w+\s*=\s*'(?:[^']*)'", '', cleaned)
    # remove javascript: URIs
    cleaned = re.sub(r'(?i)javascript:', '', cleaned)
    return cleaned


def _render_signature_to_html(signature: str) -> str:
    signature = signature or ""
    # if it looks like HTML, sanitize and return; otherwise translate simple markdown/plain to HTML
    if "<" in signature and ">" in signature:
        sig_html = _sanitize_signature_html(signature)
        return sig_html
    else:
        return _simple_markdown_to_html(signature)


def build_personal_html(
    body_template: str,
    signature: str,
    smtp_email: str,
    sender_name: str,
    emp_name: str,
):
    """
    Build the HTML alternative by converting the personalized plain body to a simple HTML.
    This returns the full HTML (body + signature). For preview we prefer using helper
    _prepare_preview_body_and_signature to keep body and signature separate.
    """
    plain = build_personal_body(
        body_template=body_template,
        signature=signature,
        smtp_email=smtp_email,
        sender_name=sender_name,
        emp_name=emp_name,
    )
    html = _simple_markdown_to_html(plain)
    return html


def _prepare_preview_body_and_signature(body_template, signature, smtp_email, sender_name, emp_name):
    """
    Return (plain_body, body_html_only, signature_html)
    body_html_only does NOT include signature_html.
    """
    body_template = (body_template or "").strip()
    if not body_template:
        body_template = "Hi {name},\n\nPlease see the details below."

    full_name = str(emp_name or "").strip()
    first_name = full_name.split()[0] if full_name else ""
    try:
        body = body_template.format(name=first_name, full_name=full_name)
    except Exception:
        try:
            body = body_template.replace("{name}", first_name).replace("{full_name}", full_name)
        except Exception:
            body = body_template

    plain_body = body
    # Convert body to HTML (body only)
    html_body_only = _simple_markdown_to_html(body)
    signature_html = _render_signature_to_html(signature)
    return plain_body, html_body_only, signature_html


def send_email_to_employee(
    server,
    to_email: str,
    emp_name: str,
    subject: str,
    body_template: str,
    signature: str,
    smtp_email: str,
    sender_name: str,
    attachments: List[Tuple[str, bytes]],
    cc_list: List[str],
    bcc_list: List[str],
) -> str:
    if not to_email or str(to_email).strip() == "":
        return "Error: No email address"
    to_email = str(to_email).strip()
    if not is_valid_email(to_email):
        return "Error: Invalid email"
    msg = EmailMessage()
    msg["Subject"] = sanitize_header_value(subject or "Notification")
    from_name_clean = sanitize_header_value(sender_name)
    msg["From"] = f"{from_name_clean} <{smtp_email}>"
    msg["To"] = to_email
    if cc_list:
        msg["Cc"] = ", ".join([sanitize_header_value(c) for c in cc_list if c])

    # plain text content contains signature appended (so plain fallback works)
    full_body = build_personal_body(
        body_template=body_template,
        signature=signature,
        smtp_email=smtp_email,
        sender_name=sender_name,
        emp_name=str(emp_name or ""),
    )
    msg.set_content(full_body)

    # HTML alternative: build body-only html + signature html and combine (ensures single signature)
    try:
        plain_body, html_body_only, signature_html = _prepare_preview_body_and_signature(
            body_template=body_template,
            signature=signature,
            smtp_email=smtp_email,
            sender_name=sender_name,
            emp_name=str(emp_name or ""),
        )
        full_html = html_body_only + "\n<div class=\"signature\">\n" + signature_html + "\n</div>"
        msg.add_alternative(full_html, subtype="html")
    except Exception as e:
        logger.warning("Failed to add HTML alternative: %s", e)

    for fname, fdata in attachments:
        try:
            maintype, subtype = guess_mime_type(fname)
            msg.add_attachment(fdata, maintype=maintype, subtype=subtype, filename=fname)
        except Exception as e:
            logger.warning("Attachment add failed for %s: %s", fname, e)
            msg.add_attachment(fdata, maintype="application", subtype="octet-stream", filename=fname)

    all_recipients = [to_email] + [c for c in cc_list if c] + [b for b in bcc_list if b]
    try:
        server.send_message(msg, from_addr=smtp_email, to_addrs=all_recipients)
        return "OK"
    except smtplib.SMTPRecipientsRefused as e:
        logger.exception("Recipients refused for %s: %s", to_email, e)
        return f"Error: RecipientsRefused {e}"
    except smtplib.SMTPDataError as e:
        logger.exception("SMTP data error for %s: %s", to_email, e)
        return f"Error: SMTPDataError {e}"
    except smtplib.SMTPServerDisconnected as e:
        logger.exception("Server disconnected while sending to %s: %s", to_email, e)
        return f"Error: ServerDisconnected {e}"
    except Exception as e:
        logger.exception("Send failed for %s: %s", to_email, e)
        return f"Error: {e}"


# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prefilled_signature = ""
    query_smtp = request.args.get("smtp_email", "").strip()
    if query_smtp:
        prefilled_signature = get_saved_signature(query_smtp) or ""

    if request.method == "POST":
        purge_old_reports()
        # If confirm_send is not present, block real sending and ask user to preview first.
        confirm_send = request.form.get("confirm_send", "") == "1"

        smtp_email = request.form.get("smtp_email", "").strip()
        smtp_pass = request.form.get("smtp_pass", "").strip()
        sender_name = request.form.get("sender_name", "").strip() or APP_NAME
        skip_if_no_pdf = request.form.get("skip_if_no_pdf") == "on"
        use_saved_signature_flag = request.form.get("use_saved_signature") == "on"
        save_default_signature_flag = request.form.get("save_default_signature") == "on"
        signature_from_form = request.form.get("signature", "") or ""

        # If user attempted to POST without confirming (e.g. direct submit), block it and instruct to preview.
        if not confirm_send:
            flash("Please use the Preview → Confirm flow before sending. Click 'Send Emails' to preview, then 'Send All' in the preview modal to confirm.", "error")
            # Re-render the form (do not proceed to send)
            return render_template_safe(
                "send_email.html",
                prefilled_signature=prefilled_signature,
                saved_signature=get_saved_signature(smtp_email) or "",
                request_form=request.form,
            )

        if use_saved_signature_flag:
            saved = get_saved_signature(smtp_email)
            signature_to_use = saved if saved else signature_from_form
        else:
            signature_to_use = signature_from_form

        if save_default_signature_flag and smtp_email:
            try:
                save_signature_for_email(smtp_email, signature_from_form)
                flash("Saved signature as default for this SMTP email.", "success")
            except Exception:
                flash("Could not save signature (server error).", "error")

        if not smtp_email or not smtp_pass:
            flash("Please enter SMTP Email (Gmail) and App Password.", "error")
            return redirect(url_for("index"))

        excel_file = request.files.get("employees_file")
        if not excel_file or excel_file.filename == "":
            flash("Please upload Employees Excel (.xlsx or .csv).", "error")
            return redirect(url_for("index"))

        pdf_files = request.files.getlist("pdf_files")
        pdf_store = []
        for f in pdf_files:
            if f and f.filename:
                try:
                    pdf_store.append((f.filename, f.read()))
                except Exception as e:
                    logger.warning("Could not read uploaded file %s: %s", f.filename, e)

        subject = request.form.get("subject", "").strip() or "Notification"
        body_template = request.form.get("body", "")

        cc_raw = request.form.get("cc", "")
        bcc_raw = request.form.get("bcc", "")
        cc_list = [x.strip() for x in cc_raw.split(",") if x.strip()]
        bcc_list = [x.strip() for x in bcc_raw.split(",") if x.strip()]

        try:
            filename_lower = excel_file.filename.lower()
            if filename_lower.endswith(".csv"):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file)
        except Exception as e:
            logger.exception("Error reading employees file: %s", e)
            flash(f"Error reading employees file: {e}", "error")
            return redirect(url_for("index"))

        email_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("email", "email_id", "mail"):
                email_col = col
                break
        if email_col is None:
            flash("Employees file must contain an 'email' column (named email / email_id / mail).", "error")
            return redirect(url_for("index"))

        name_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("name", "emp_name", "employee_name"):
                name_col = col
                break

        total = len(df)
        sent = 0
        errors = 0
        start_time = time.time()
        report_rows = []
        use_pdf_matching = bool(pdf_store) and (name_col is not None)

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
                server.starttls()
                server.login(smtp_email, smtp_pass)

                for idx, row in df.iterrows():
                    to_email = row.get(email_col, "")
                    emp_name = row.get(name_col, "") if name_col else ""

                    if not is_valid_email(to_email):
                        errors += 1
                        status_str = "Error: Invalid email"
                        report_rows.append({"emp_name": str(emp_name or ""), "email": str(to_email or ""), "status": status_str, "pdf_attached": False, "skipped_reason": "invalid_email"})
                        continue

                    attachments_for_emp = []
                    pdf_attached_flag = False
                    if use_pdf_matching:
                        matched = find_matching_pdf(emp_name, pdf_store)
                        if matched:
                            attachments_for_emp = [matched]
                            pdf_attached_flag = True
                        else:
                            if skip_if_no_pdf:
                                errors += 1
                                status_str = "Skipped: No matching PDF for employee"
                                report_rows.append({"emp_name": str(emp_name or ""), "email": str(to_email or ""), "status": status_str, "pdf_attached": False, "skipped_reason": "pdf_missing"})
                                continue
                            else:
                                attachments_for_emp = []
                                pdf_attached_flag = False

                    try:
                        status = send_email_to_employee(
                            server=server,
                            to_email=to_email,
                            emp_name=emp_name,
                            subject=subject,
                            body_template=body_template,
                            signature=signature_to_use,
                            smtp_email=smtp_email,
                            sender_name=sender_name,
                            attachments=attachments_for_emp,
                            cc_list=cc_list,
                            bcc_list=bcc_list,
                        )
                        if status == "OK":
                            sent += 1
                            status_str = "Sent"
                        else:
                            errors += 1
                            status_str = status
                    except Exception as inner_e:
                        logger.exception("Error sending to %s: %s", to_email, inner_e)
                        errors += 1
                        status_str = f"Error: {inner_e}"

                    report_rows.append({"emp_name": str(emp_name or ""), "email": str(to_email or ""), "status": status_str, "pdf_attached": pdf_attached_flag, "skipped_reason": None})

            elapsed = round(time.time() - start_time, 2)
            summary = {"total": total, "sent": sent, "errors": errors, "time_taken": f"{elapsed} sec"}

            if errors == 0:
                flash(f"Emails sent successfully to {sent} / {total} recipients in {elapsed} seconds.", "success")
            else:
                flash(f"Completed with some errors. Sent: {sent} / {total}, Errors: {errors}, Time: {elapsed} seconds.", "error")

            report_id = str(uuid.uuid4())
            GENERATED_REPORTS[report_id] = {"rows": report_rows, "ts": time.time()}

            purge_old_reports()

            saved_for_account = get_saved_signature(smtp_email)
            return render_template_safe(
                "send_email.html",
                summary=summary,
                report_id=report_id,
                report_rows=report_rows,
                saved_signature=saved_for_account or "",
                prefilled_signature=get_saved_signature(smtp_email) or "",
            )

        except smtplib.SMTPAuthenticationError:
            logger.exception("SMTP auth failed")
            flash("SMTP login failed. Please check your Gmail ID or App Password.", "error")
        except Exception as e:
            logger.exception("Error while sending emails: %s", e)
            flash(f"Error while sending emails: {e}", "error")

        return redirect(url_for("index"))

    # GET
    return render_template_safe("send_email.html", prefilled_signature=prefilled_signature, saved_signature=get_saved_signature(query_smtp) or "")


@app.route("/preview", methods=["POST"])
def preview():
    """
    Returns a JSON preview for first N recipients including:
      - email fields: to, cc, bcc, subject
      - body (plain), html (body-only), signature_html (separate)
      - attachments: list of filenames (matching pdfs)
    """
    try:
        smtp_email = request.form.get("smtp_email", "").strip()
        sender_name = request.form.get("sender_name", "").strip() or APP_NAME
        signature_from_form = request.form.get("signature", "") or ""
        use_saved_signature_flag = request.form.get("use_saved_signature") == "on"
        if use_saved_signature_flag and smtp_email:
            signature_to_use = get_saved_signature(smtp_email) or signature_from_form
        else:
            signature_to_use = signature_from_form

        subject = request.form.get("subject", "").strip() or "Notification"
        body_template = request.form.get("body", "") or ""
        try:
            max_preview = int(request.form.get("max_preview", 3))
        except Exception:
            max_preview = 3
        max_preview = max(1, min(10, max_preview))

        # read uploaded employees file
        excel_file = request.files.get("employees_file")
        if not excel_file:
            return jsonify({"ok": False, "error": "No employees file uploaded"}), 400

        # read uploaded pdf files (for matching) if any
        pdf_files = request.files.getlist("pdf_files")
        pdf_store: List[Tuple[str, bytes]] = []
        for f in pdf_files:
            if f and f.filename:
                try:
                    pdf_store.append((f.filename, f.read()))
                except Exception as e:
                    logger.warning("Could not read uploaded file %s: %s", f.filename, e)

        # parse employee file
        filename_lower = excel_file.filename.lower()
        try:
            if filename_lower.endswith(".csv"):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file)
        except Exception as e:
            logger.exception("Error reading employees file for preview: %s", e)
            return jsonify({"ok": False, "error": f"Could not read file: {e}"}), 400

        email_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("email", "email_id", "mail"):
                email_col = col
                break
        if email_col is None:
            return jsonify({"ok": False, "error": "Employees file must contain an 'email' column (email / email_id / mail)"}), 400

        name_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("name", "emp_name", "employee_name"):
                name_col = col
                break

        # parse CC/BCC inputs from form
        cc_raw = request.form.get("cc", "")
        bcc_raw = request.form.get("bcc", "")
        cc_list = [x.strip() for x in cc_raw.split(",") if x.strip()]
        bcc_list = [x.strip() for x in bcc_raw.split(",") if x.strip()]

        previews = []
        for idx, row in df.iterrows():
            if len(previews) >= max_preview:
                break
            to_email = row.get(email_col, "")
            emp_name = row.get(name_col, "") if name_col else ""

            if not is_valid_email(to_email):
                previews.append({
                    "index": int(idx),
                    "emp_name": str(emp_name or ""),
                    "email": str(to_email or ""),
                    "status": "Invalid email",
                    "subject": subject,
                    "body": "",
                    "html": "",
                    "cc": cc_list,
                    "bcc": bcc_list,
                    "signature": signature_to_use,
                    "signature_html": _render_signature_to_html(signature_to_use),
                    "attachments": [],
                })
                continue

            # Use helper that returns body-only HTML and signature separately
            plain_body, html_body_only, signature_html = _prepare_preview_body_and_signature(
                body_template=body_template,
                signature=signature_to_use,
                smtp_email=smtp_email,
                sender_name=sender_name,
                emp_name=str(emp_name or ""),
            )

            # find matching pdf filename(s) for this row (if any)
            attachments = []
            if pdf_store and name_col:
                matched = find_matching_pdf(emp_name, pdf_store)
                if matched:
                    fname, _ = matched
                    attachments.append(fname)

            previews.append({
                "index": int(idx),
                "emp_name": str(emp_name or ""),
                "to": str(to_email or ""),
                "subject": subject,
                "body": plain_body,
                # html contains body-only (no signature)
                "html": html_body_only,
                "status": "OK",
                "cc": cc_list,
                "bcc": bcc_list,
                "signature": signature_to_use,
                "signature_html": signature_html,
                "attachments": attachments,
            })

        return jsonify({"ok": True, "previews": previews})

    except Exception as e:
        logger.exception("Preview error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/send_single", methods=["POST"])
def send_single():
    """
    Endpoint to send only the first valid recipient from the uploaded employees file.
    """
    try:
        purge_old_reports()
        smtp_email = request.form.get("smtp_email", "").strip()
        smtp_pass = request.form.get("smtp_pass", "").strip()
        sender_name = request.form.get("sender_name", "").strip() or APP_NAME
        skip_if_no_pdf = request.form.get("skip_if_no_pdf") == "on"
        use_saved_signature_flag = request.form.get("use_saved_signature") == "on"
        save_default_signature_flag = request.form.get("save_default_signature") == "on"
        signature_from_form = request.form.get("signature", "") or ""

        if use_saved_signature_flag:
            saved = get_saved_signature(smtp_email)
            signature_to_use = saved if saved else signature_from_form
        else:
            signature_to_use = signature_from_form

        if save_default_signature_flag and smtp_email:
            try:
                save_signature_for_email(smtp_email, signature_from_form)
                flash("Saved signature as default for this SMTP email.", "success")
            except Exception:
                flash("Could not save signature (server error).", "error")

        if not smtp_email or not smtp_pass:
            flash("Please enter SMTP Email (Gmail) and App Password.", "error")
            return redirect(url_for("index"))

        excel_file = request.files.get("employees_file")
        if not excel_file or excel_file.filename == "":
            flash("Please upload Employees Excel (.xlsx or .csv).", "error")
            return redirect(url_for("index"))

        pdf_files = request.files.getlist("pdf_files")
        pdf_store = []
        for f in pdf_files:
            if f and f.filename:
                try:
                    pdf_store.append((f.filename, f.read()))
                except Exception as e:
                    logger.warning("Could not read uploaded file %s: %s", f.filename, e)

        subject = request.form.get("subject", "").strip() or "Notification"
        body_template = request.form.get("body", "")

        cc_raw = request.form.get("cc", "")
        bcc_raw = request.form.get("bcc", "")
        cc_list = [x.strip() for x in cc_raw.split(",") if x.strip()]
        bcc_list = [x.strip() for x in bcc_raw.split(",") if x.strip()]

        # read employees file
        try:
            filename_lower = excel_file.filename.lower()
            if filename_lower.endswith(".csv"):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file)
        except Exception as e:
            logger.exception("Error reading employees file for send_single: %s", e)
            flash(f"Error reading employees file: {e}", "error")
            return redirect(url_for("index"))

        email_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("email", "email_id", "mail"):
                email_col = col
                break
        if email_col is None:
            flash("Employees file must contain an 'email' column (named email / email_id / mail).", "error")
            return redirect(url_for("index"))

        name_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("name", "emp_name", "employee_name"):
                name_col = col
                break

        # find the first valid email row
        first_row = None
        for idx, row in df.iterrows():
            to_email = row.get(email_col, "")
            if is_valid_email(to_email):
                first_row = row
                break

        if first_row is None:
            flash("No valid recipient found in the uploaded file.", "error")
            return redirect(url_for("index"))

        emp_name = first_row.get(name_col, "") if name_col else ""
        to_email = first_row.get(email_col, "")

        attachments_for_emp = []
        use_pdf_matching = bool(pdf_store) and (name_col is not None)
        if use_pdf_matching:
            matched = find_matching_pdf(emp_name, pdf_store)
            if matched:
                attachments_for_emp = [matched]
            else:
                if skip_if_no_pdf:
                    flash("Skipped: No matching PDF found for the first recipient (skip_if_no_pdf is enabled).", "error")
                    return redirect(url_for("index"))
                attachments_for_emp = []

        sent = 0
        errors = 0
        report_rows = []
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
                server.starttls()
                server.login(smtp_email, smtp_pass)

                status = send_email_to_employee(
                    server=server,
                    to_email=to_email,
                    emp_name=emp_name,
                    subject=subject,
                    body_template=body_template,
                    signature=signature_to_use,
                    smtp_email=smtp_email,
                    sender_name=sender_name,
                    attachments=attachments_for_emp,
                    cc_list=cc_list,
                    bcc_list=bcc_list,
                )
                if status == "OK":
                    sent = 1
                    status_str = "Sent"
                else:
                    errors = 1
                    status_str = status

                report_rows.append({"emp_name": str(emp_name or ""), "email": str(to_email or ""), "status": status_str, "pdf_attached": bool(attachments_for_emp), "skipped_reason": None})

            summary = {"total": 1, "sent": sent, "errors": errors, "time_taken": "single-send"}
            flash(f"Send single completed. Status: {status_str}", "success" if status_str == "Sent" else "error")

            report_id = str(uuid.uuid4())
            GENERATED_REPORTS[report_id] = {"rows": report_rows, "ts": time.time()}
            purge_old_reports()
            return render_template_safe(
                "send_email.html",
                summary=summary,
                report_id=report_id,
                report_rows=report_rows,
                saved_signature=get_saved_signature(smtp_email) or "",
                prefilled_signature=get_saved_signature(smtp_email) or "",
            )
        except smtplib.SMTPAuthenticationError:
            logger.exception("SMTP auth failed (single send)")
            flash("SMTP login failed. Please check your Gmail ID or App Password.", "error")
        except Exception as e:
            logger.exception("Error while sending single email: %s", e)
            flash(f"Error while sending single email: {e}", "error")

        return redirect(url_for("index"))
    except Exception as e:
        logger.exception("send_single handler failed: %s", e)
        flash(f"Server error: {e}", "error")
        return redirect(url_for("index"))


@app.route("/download_report/<report_id>")
def download_report(report_id):
    report = GENERATED_REPORTS.get(report_id)
    if report is None:
        flash("Report not found or expired.", "error")
        return redirect(url_for("index"))

    rows = report["rows"]
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    engines_to_try = ["openpyxl", "xlsxwriter"]
    excel_written = False
    last_exc = None

    for engine in engines_to_try:
        try:
            output.seek(0)
            output.truncate(0)
            with pd.ExcelWriter(output, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name="Email Report")
            excel_written = True
            logger.debug("Wrote excel with engine %s", engine)
            break
        except Exception as e:
            last_exc = e
            logger.exception("Could not write Excel with engine %s: %s", engine, e)

    try:
        ts = report.get("ts", time.time())
        timestr = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        if excel_written:
            output.seek(0)
            data = output.getvalue()
            filename = f"email_report_{timestr}.xlsx"
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(data)),
            }
            return Response(data, mimetype=mimetype, headers=headers)
        else:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            filename = f"email_report_{timestr}.csv"
            mimetype = "text/csv; charset=utf-8"
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(csv_bytes)),
            }
            return Response(csv_bytes, mimetype=mimetype, headers=headers)
    except Exception as final_e:
        logger.exception("Failed to prepare/send report download for %s: %s (last_exc=%s)", report_id, final_e, last_exc)
        flash("Could not prepare report download (server error). Check server logs.", "error")
        return redirect(url_for("index"))


@app.errorhandler(500)
def internal_server_error(e):
    logger.exception("Unhandled exception: %s", e)
    try:
        return render_template_safe("500.html", error=str(e)), 500
    except Exception:
        return ("<h2>Internal Server Error</h2><p>Check server logs for details.</p>"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
