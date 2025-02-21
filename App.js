// File: src/App.js
import React, { useState } from 'react';
import { AppBar, Tabs, Tab, Box, Container } from '@mui/material';
import RawEchogram from './components/RawEchogram';
import TrackedEchogram from './components/TrackedEchogram';
import SpatialPlot from './components/SpatialPlot';
import TrackSummary from './components/TrackSummary';
// IMPORTANT: The file in src/components is named "TrainingVisualizations.js" (exactly)
import TrainingVisualizations from './components/TrainingVisualizations';

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabIndex, setTabIndex] = useState(0);

  const handleChange = (event, newValue) => {
    setTabIndex(newValue);
  };

  return (
    <>
      <AppBar position="static">
        <Tabs
          value={tabIndex}
          onChange={handleChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Raw Echogram" />
          <Tab label="Tracked Echogram" />
          <Tab label="3D Spatial Plot" />
          <Tab label="Track Summary" />
          <Tab label="Training Visualizations" />
        </Tabs>
      </AppBar>
      <Container sx={{ mt: 4 }}>
        <TabPanel value={tabIndex} index={0}>
          <RawEchogram />
        </TabPanel>
        <TabPanel value={tabIndex} index={1}>
          <TrackedEchogram />
        </TabPanel>
        <TabPanel value={tabIndex} index={2}>
          <SpatialPlot />
        </TabPanel>
        <TabPanel value={tabIndex} index={3}>
          <TrackSummary />
        </TabPanel>
        <TabPanel value={tabIndex} index={4}>
          <TrainingVisualizations />
        </TabPanel>
      </Container>
    </>
  );
}

export default App;
