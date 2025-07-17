# SecureTime Pro

Employee time tracking system using facial recognition. Look at the camera for 2 seconds to automatically clock in/out.

## What it does

- **Face recognition** identifies employees automatically
- **Auto clock in/out** - no buttons to press
- **Liveness detection** prevents photo spoofing
- **Time tracking** calculates work hours
- **Web interface** for management and reports

## Installation

### Requirements
- Python 3.8+
- Webcam
- Modern browser (Chrome recommended)

### Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/securetime-pro.git
   cd securetime-pro
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run application**
   ```bash
   python face_recognition_webui.py
   ```

4. **Open browser**
   ```
   http://localhost:5000
   ```

## How to use

### Add employees
1. Go to "Add Employee"
2. Enter name
3. Take photos from different angles (front, left, right)
4. System trains recognition model

### Time tracking
1. Go to "Time Clock"  
2. Click "Start Camera"
3. Look at camera for 2 seconds
4. System automatically:
   - Clocks you **IN** if you're not working
   - Clocks you **OUT** if you're currently working
5. Camera stops automatically

### View reports
1. Go to "Reports"
2. Select date range
3. Generate time reports

## How it works

### Face Recognition
- Uses `dlib` and `face_recognition` libraries
- Adaptive learning improves accuracy over time
- Confidence threshold prevents false matches

### Liveness Detection
- Detects real faces vs photos
- Analyzes face movement and quality
- Prevents spoofing with printed photos

### Database
- SQLite stores employee data and time records
- Each work session = one record with clock_in/clock_out times
- Automatic hours calculation

### Auto Clock Logic
```
Current Status → Action
Not clocked in → Clock IN
Clocked in → Clock OUT
Previously clocked out → Clock IN (new session)
```

## File Structure

```
SecureTime-Pro/
├── face_recognition_webui.py          # Main Flask application
├── maml_face_recognition.py           # Face recognition engine  
├── anti_spoofing_system.py            # Liveness detection
├── improved_few_shot.py               # Learning algorithms
├── face_recognition_system.py         # Base recognition system
├── templates/                         # HTML templates
│   ├── base.html                      # Main layout
│   ├── timeclock.html                 # Time clock interface
│   ├── index.html                     # Dashboard
│   ├── employees.html                 # Employee list
│   ├── add_employee.html              # Add new employee
│   └── reports.html                   # Time reports
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Configuration

Edit these values in `face_recognition_webui.py`:

```python
RECOGNITION_THRESHOLD = 0.6          # Minimum confidence to recognize (60%)
CLOCK_ACTION_THRESHOLD = 60          # Minimum confidence to clock in/out (60%)
AUTO_CLOCK_DELAY = 2                 # Seconds to hold steady (2 seconds)
```

## API Endpoints

### REST API
- `GET /api/status` - System status
- `GET /api/employees` - List employees
- `GET /api/time_records` - Get time records
- `DELETE /api/delete_employee/<name>` - Remove employee

### WebSocket Events
- `start_session` - Start camera
- `video_frame` - Process video frame
- `auto_clock_result` - Clock action result

## Troubleshooting

**Camera not working:**
- Check browser permissions
- Close other apps using camera
- Try different browser

**Recognition not accurate:**
- Add more training photos
- Ensure good lighting
- Check camera focus

**Database errors:**
- Check file permissions
- Restart application

## Development

### Debug Mode
Add debug routes to see employee status:
```python
@app.route('/api/debug/<employee_name>')
def debug_employee(employee_name):
    # Shows current status and all records
```

### Database Schema

**employees table:**
- `id`, `name`, `department`, `created_at`

**time_records table:**
- `id`, `employee_name`, `clock_in`, `clock_out`, `hours_worked`, `date`, `confidence`

## Security Notes

- Employee photos stored locally only
- Database contains sensitive time data
- Use proper access controls in production
- Consider encryption for sensitive environments

## License

MIT License - see LICENSE file

## Technology Stack

- **Backend:** Flask, Flask-SocketIO
- **Face Recognition:** dlib, face_recognition, OpenCV
- **Database:** SQLite
- **Frontend:** HTML5, JavaScript, WebSocket
- **ML:** scikit-learn, numpy