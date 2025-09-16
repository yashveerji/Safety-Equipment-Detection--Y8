import React from 'react';
import { ControlsPanel } from './ControlsPanel';
import { StreamPanel } from './StreamPanel';
import { StatsPanel } from './StatsPanel';
import { AlertsList } from './AlertsList';
import { DetectionTable } from './DetectionTable';
import { ZonesPanel } from './ZonesPanel';

export const Dashboard: React.FC = () => {
  return (
    <div className="grid gap-8 xl:grid-cols-3">
      <div className="xl:col-span-2 space-y-8">
        <StreamPanel />
        <DetectionTable />
        <AlertsList />
      </div>
      <div className="space-y-8">
        <ControlsPanel />
        <StatsPanel />
        <ZonesPanel />
      </div>
    </div>
  );
};
