# Frontend - AI Waste Segregation

React + Vite frontend for waste classification application.

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx              # Entry point
    ├── App.jsx               # Main app component
    ├── api/
    │   └── client.js         # API client (axios)
    ├── components/
    │   ├── Navbar.jsx
    │   ├── Navbar.css
    │   ├── WasteClassifier.jsx
    │   ├── WasteClassifier.css
    │   ├── PredictionResult.jsx
    │   ├── PredictionResult.css
    │   ├── HistoryList.jsx
    │   └── HistoryList.css
    └── styles/
        └── App.css
```

## Features

### 🎯 WasteClassifier Component
- **Upload Mode**: Select images from disk
- **Webcam Mode**: Capture photos directly from camera
- Image preview before classification
- Loading states during API calls
- Error handling with user-friendly messages

### 📊 PredictionResult Component
- Display predicted waste category with icon
- Visual confidence scores for all categories
- Color-coded probability bars
- Responsive design

### 📜 HistoryList Component
- Shows last 10 predictions
- Image thumbnails with class and timestamp
- Hover effects for better UX

## Configuration

### Backend URL

The frontend proxies API requests to the backend. Configure in `vite.config.js`:

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

For production, set the `VITE_API_URL` environment variable:

```bash
VITE_API_URL=https://your-backend-url.com npm run build
```

## Building for Production

### 1. Build

```bash
npm run build
```

This creates optimized files in the `dist/` folder.

### 2. Preview

```bash
npm run preview
```

### 3. Deploy

Upload the contents of `dist/` to your hosting provider (Netlify, Vercel, etc.)

## API Integration

The app communicates with the backend via REST API:

### Endpoints Used

1. **Health Check** (optional)
   ```javascript
   GET /api/health
   ```

2. **Predict**
   ```javascript
   POST /api/predict
   Content-Type: multipart/form-data
   Body: { file: File }
   ```

### API Client

Located in `src/api/client.js`:

```javascript
import { predictWaste, checkHealth } from './api/client';

// Usage
const result = await predictWaste(imageFile);
console.log(result.predicted_class);
console.log(result.probabilities);
```

## Styling

The app uses vanilla CSS with:
- Responsive grid layouts
- Gradient backgrounds
- Smooth transitions and hover effects
- Mobile-first approach

### Color Scheme

- **Organic**: Green (#4caf50)
- **Plastic**: Blue (#2196f3)
- **Paper**: Orange (#ff9800)
- **Metal**: Gray (#9e9e9e)

## Browser Support

- Modern browsers with ES6+ support
- Camera access requires HTTPS (except localhost)

## Webcam Permissions

The webcam feature requires camera permissions:
1. Browser will prompt for permission on first use
2. Must be served over HTTPS in production
3. Localhost automatically allowed during development

## Troubleshooting

### API Connection Issues

If you see "Network error" messages:
1. Ensure the backend is running on `http://localhost:8000`
2. Check CORS settings in the backend
3. Verify the proxy configuration in `vite.config.js`

### Webcam Not Working

1. Grant camera permissions in browser
2. Check if camera is in use by another application
3. Ensure HTTPS is used (or localhost for dev)

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Development Tips

### Hot Module Replacement

Vite provides instant HMR. Save any file and see changes immediately without full page reload.

### Component Development

Each component is self-contained with its own CSS file for easier maintenance.

### Adding New Features

1. Create component in `src/components/`
2. Import and use in `App.jsx`
3. Add corresponding CSS file
4. Update API client if needed

## Performance

- Lazy loading for images
- Optimized bundle size with Vite
- Efficient re-renders with React
- Debounced API calls (can be added if needed)
