# 📧 LCF MailDesk  
Bulk Email Sender for Lighthouse Communities Foundation

LCF MailDesk is a Flask-based bulk email sending tool that allows you to:

- Upload Excel/CSV files with employee email lists  
- Upload optional per-employee PDF attachments  
- Use custom placeholders like `{name}` and `{full_name}`  
- Preview emails before sending  
- Send all emails or send only the first email for testing  
- Download Excel reports of delivery results  
- Save/load email signatures linked to each SMTP email  

---

## 🚀 Features

### ✔ Bulk email sending  
Upload an employee sheet (`.xlsx` or `.csv`) and send personalized emails.

### ✔ Automatic PDF matching  
Attach per-employee PDFs based on filename similarity.

### ✔ Smart placeholders  
Use:
- `{name}` → First name only  
- `{full_name}` → Complete employee name  

### ✔ Email Preview  
See how your first email will look before sending.

### ✔ Signature Storage  
Each SMTP email can store a default signature.

### ✔ Delivery Report  
Download a timestamped Excel report with success/error status.

---

## 📂 Project Structure

