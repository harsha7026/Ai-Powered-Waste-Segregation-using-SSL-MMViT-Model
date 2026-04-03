import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const predictWaste = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await apiClient.post('/api/predict', formData);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Prediction failed');
    } else {
      throw new Error('Network error. Please check if the backend is running.');
    }
  }
};

export const generateGradCam = async (file, targetClassIndex = null) => {
  const formData = new FormData();
  formData.append('file', file);

  if (targetClassIndex !== null && targetClassIndex !== undefined) {
    formData.append('target_class_idx', targetClassIndex);
  }

  try {
    const response = await apiClient.post('/api/grad-cam', formData);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Grad-CAM generation failed');
    }
    throw new Error('Network error. Could not generate Grad-CAM visualization.');
  }
};

export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/api/health');
    return response.data;
  } catch (error) {
    throw new Error('Backend health check failed');
  }
};

export const getStatsSummary = async () => {
  try {
    const response = await apiClient.get('/api/stats/summary');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load summary statistics');
  }
};

export const getClassDistribution = async () => {
  try {
    const response = await apiClient.get('/api/stats/class-distribution');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load class distribution');
  }
};

export const getDisposalRules = async () => {
  try {
    const response = await apiClient.get('/api/admin/disposal-rules');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load disposal rules');
  }
};

export const saveDisposalRules = async (rules) => {
  try {
    const response = await apiClient.post('/api/admin/disposal-rules', rules);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to save disposal rules');
    }
    throw new Error('Network error. Could not save disposal rules.');
  }
};
