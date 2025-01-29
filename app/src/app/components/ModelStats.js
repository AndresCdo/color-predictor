import React from 'react';
import PropTypes from 'prop-types';

const ModelStats = ({ stats, modelStats }) => {
  return (
    <div className="mb-6">
      <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
        <StatItem label="Liked Colors" value={stats.totalSamples - stats.unselectedCount} />
        <StatItem label="Disliked Colors" value={stats.unselectedCount} />
        <StatItem label="Total Samples" value={stats.totalSamples} />
        <StatItem label="Like Ratio" value={`${stats.likedPercentage}%`} />
      </div>
      {modelStats.lastTrained && (
        <div className="text-sm text-gray-600">
          <p>Last trained: {modelStats.lastTrained}</p>
          <p>Accuracy: {modelStats.accuracy}%</p>
        </div>
      )}
    </div>
  );
};

const StatItem = ({ label, value }) => (
  <div className="flex justify-between">
    <span>{label}:</span>
    <span className="font-medium">{value}</span>
  </div>
);

StatItem.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]).isRequired,
};

ModelStats.propTypes = {
  stats: PropTypes.shape({
    totalSamples: PropTypes.number.isRequired,
    unselectedCount: PropTypes.number.isRequired,
    likedPercentage: PropTypes.string.isRequired,
  }).isRequired,
  modelStats: PropTypes.shape({
    lastTrained: PropTypes.string,
    accuracy: PropTypes.string,
  }).isRequired,
};

export default ModelStats;
