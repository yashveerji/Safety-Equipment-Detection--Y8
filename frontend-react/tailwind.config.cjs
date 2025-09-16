/**** Tailwind configuration for PPE Vision Console ****/
module.exports = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}'
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0b1118',
        panel: '#16212c',
        panelAlt: '#1d2a36',
        accent: '#2d7dff',
        accentB: '#6a5af9',
        danger: '#ff4d5b',
        ok: '#18c27c',
        warn: '#ffb547'
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui']
      },
      boxShadow: {
        panel: '0 4px 18px -4px rgba(0,0,0,0.55),0 6px 32px -8px rgba(0,0,0,0.55)'
      }
    }
  },
  plugins: []
};
