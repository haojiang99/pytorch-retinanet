# RetinaNet Mammography Client

This is a Svelte.js client application for the mammogram analysis tool. It provides a user interface for uploading mammogram images and viewing the analysis results.

## Setup

1. Install the dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:3000

3. For production:

```bash
npm run build
```

This will create a `dist` folder with the compiled application.

## Application Structure

- `src/App.svelte`: Main application component
- `src/routes/Home.svelte`: Upload page with drag-and-drop functionality
- `src/routes/Results.svelte`: Results page showing detection results and analysis

## Features

- Drag-and-drop image upload
- Real-time mammogram analysis
- Visualization of detected masses with bounding boxes
- Detailed analysis summary with classification information
- Support for JPEG and PNG images

## Integration with Server

This client communicates with the Flask server which runs the RetinaNet model. Make sure the server is running at http://localhost:5000 before using the client.

You can modify the API URL in the `Home.svelte` component if needed.
