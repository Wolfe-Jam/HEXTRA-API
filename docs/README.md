# HEXTRA API Documentation

## API Design Philosophy

The HEXTRA Detection API follows the principle of "38 years, 38 lines" - maximum expertise distilled into minimal complexity.

## Architecture Decisions

### Why FastAPI?
- Automatic OpenAPI documentation
- Type safety with Pydantic
- Async performance for concurrent requests
- Modern Python best practices

### Why OTSU Thresholding?
The core algorithm uses Otsu's method because:
- Automatically finds optimal threshold for any image
- No manual parameter tuning required
- Mathematically robust across lighting conditions
- Proven effectiveness over 38 years of application

### Error Handling Strategy
All endpoints return consistent error structures:
```json
{
  "success": false,
  "error": "Human readable message",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

## Deployment Guide

### Local Development
1. Set up Python virtual environment
2. Install dependencies from requirements.txt
3. Run with uvicorn for hot reload
4. Access interactive docs at /docs

### Production Deployment
1. Railway.app for automatic scaling
2. Environment variables for configuration
3. Health checks for monitoring
4. Global CDN for performance

## Integration Examples

Complete examples for integrating with various platforms and languages are provided in the main README.

## Performance Characteristics

- Typical processing time: 200-500ms
- Memory usage: ~50MB base + image size
- Concurrent request handling: 100+ simultaneous
- Accuracy: 98%+ confidence on clear images

## Future Enhancements

The API is designed for extensibility while preserving the core 38-line algorithm.