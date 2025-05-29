# YOLOv5 Live Detection Web App

This is a web application that uses YOLOv5 for real-time object detection through your webcam.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your YOLOv5 model in the `models` directory:
```bash
mkdir -p models
cp /path/to/your/model.pt models/best.pt
```

3. Run the application:
```bash
python3 app.py
```

4. Open your browser and go to:
```
http://localhost:5000
```

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```

## Important Notes

- The application requires access to your webcam
- Make sure your browser allows camera access
- The model file should be placed in the `models` directory
- For local development, you need YOLOv5 installed in your home directory

## Health Check

You can check if the application is running properly by visiting:
```
http://localhost:5000/health
``` 