# Facial Recognition Attendance System

A simple facial recognition–based attendance system powered by **OpenCV** and **face_recognition**.  
This tool lets you:

- 📸 **Enroll students** (capture faces from webcam & register them).
- 📝 **Mark attendance** automatically by recognizing faces in real-time.
- 📂 Save attendance logs to Excel/CSV.

---

## 📦 Features

- Easy student enrollment with webcam.
- Real-time face recognition.
- Attendance automatically logged into Excel.
- Simple CLI usage (`--mode 0` for enrollment, `--mode 1` for attendance).
- Works locally — no internet required once installed.

---

## 🛠️ Installation

1. Clone or unzip the repo:
   ```bash
   git clone https://github.com/your-username/facial-recognition-attendance.git
   cd facial-recognition-attendance
   ```

   Or unzip the provided `.zip`:
   ```bash
   unzip facial-recognition-attendance-repo.zip
   cd facial-recognition-attendance-repo
   ```

2. Create and activate a virtual environment (recommended):

   **Linux/macOS**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run from the repo root:

### Enroll Students
```bash
python src/main.py --mode 0
```
- Opens webcam.
- Captures and stores student encodings.

### Mark Attendance
```bash
python src/main.py --mode 1
```
- Opens webcam.
- Recognizes registered students.
- Marks attendance into Excel (`data/attendance.xlsx`).

---

## 📂 Project Structure

```
facial-recognition-attendance/
├── src/
│   ├── face_attendance.py   # Core logic
│   ├── main.py              # CLI entry point
├── data/
│   └── students_example.xlsx # Example Excel file
├── docs/
│   └── USAGE.md             # Extra usage notes
├── tests/
│   └── test_import.py       # Basic test placeholder
├── requirements.txt
├── README.md
├── LICENSE (MIT)
└── .gitignore
```

---

## 📋 Example Excel Output

| Name      | Date       | Time     |
|-----------|-----------|----------|
| Alice     | 2025-09-09 | 09:15 AM |
| Bob       | 2025-09-09 | 09:16 AM |

---

## 🧩 Requirements

- Python 3.8+
- Webcam
- Libraries:
  - `opencv-python`
  - `face_recognition`
  - `numpy`
  - `pandas`
  - `openpyxl`

(see `requirements.txt`)

---

## ⚡ Roadmap / Ideas

- [ ] Web UI with Flask/FastAPI
- [ ] Docker support for easier setup
- [ ] CI tests (GitHub Actions)
- [ ] Export to CSV/JSON formats
- [ ] Easter egg mode 🤫

---

## 📜 License

MIT License — feel free to use, modify, and share.  
Built with ❤️ by **Blobby**.
