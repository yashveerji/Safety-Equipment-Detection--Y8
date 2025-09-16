import React from 'react';
import { Layout } from './layout/Layout';
import { Dashboard } from './dashboard/Dashboard';
import { ToastHost } from './ui/Toast';
import { AppProvider } from '../context/AppContext';

export const App: React.FC = () => {
  return (
    <AppProvider>
      <Layout>
        <Dashboard />
        <ToastHost />
      </Layout>
    </AppProvider>
  );
};
