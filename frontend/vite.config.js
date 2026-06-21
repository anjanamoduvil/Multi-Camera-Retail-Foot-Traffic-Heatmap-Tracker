import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    open: '/Multi_Camera_Retail_Foot_Traffic_Heatmap_Tracker.html'
  },
  build: {
    rollupOptions: {
      input: {
        main: 'Multi_Camera_Retail_Foot_Traffic_Heatmap_Tracker.html'
      }
    }
  }
})
