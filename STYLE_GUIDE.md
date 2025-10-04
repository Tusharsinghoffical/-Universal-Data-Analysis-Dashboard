# Universal Data Analysis Dashboard - Style Guide

This document outlines the design principles, color palette, typography, and UI components used throughout the dashboard to ensure visual consistency.

## Design Principles

### Premium Aesthetic
- Luxurious feel with gold accents and vibrant colors
- Glass-morphism effects for modern UI elements
- Smooth animations and transitions
- Consistent spacing and alignment

### Colorful & Vibrant
- Avoid black-and-white appearance
- Use gradient backgrounds
- Incorporate gold as primary accent color
- Maintain high contrast for readability

### Consistency
- Uniform design across all dashboard sections
- Consistent card styling and shadows
- Unified button and input styles
- Cohesive typography hierarchy

## Color Palette

### Primary Colors
- **Gold Accent**: `#FFD700` (Golden Yellow)
- **Secondary Gold**: `#FFA500` (Orange)
- **Vibrant Orange**: `#FF4500` (Orange Red)

### Background Gradients
- **Main Background**: `linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c)`
- **Card Background**: `rgba(255, 255, 255, 0.15)`
- **Metric Cards**: `linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))`

### Text Colors
- **Headers**: Gold gradient text
- **Body Text**: `#FFFFFF` (White)
- **Secondary Text**: `#E0E0E0` (Light Gray)

## Typography

### Font Hierarchy
- **Main Header** (Dashboard Title): 42px, Weight 800
- **Section Headers**: 28px, Weight 700
- **Sub Headers**: 24px, Weight 600
- **Card Values**: 40px, Weight 800
- **Card Labels**: 18px, Weight 500
- **Body Text**: 16px, Weight 400

### Font Styles
- **Font Family**: Default Streamlit font stack
- **Letter Spacing**: 0.5px-1px for headers
- **Text Shadows**: Subtle shadows for depth
- **Text Gradients**: Gold gradient for emphasis

## UI Components

### Cards
- **Border Radius**: 20px
- **Backdrop Filter**: `blur(12px)`
- **Box Shadow**: `0 8px 32px rgba(0,0,0,0.2)`
- **Border**: `1px solid rgba(255, 255, 255, 0.18)`
- **Hover Effect**: Scale up and elevate

### Buttons
- **Background**: `linear-gradient(45deg, #FFD700, #FFA500)`
- **Text Color**: `#1E1E1E` (Dark Gray/Black)
- **Border Radius**: 12px
- **Font Weight**: 700
- **Hover Effect**: Elevate and glow

### Tabs
- **Background**: `rgba(255, 255, 255, 0.15)`
- **Active Tab**: Gold gradient background
- **Border Radius**: 15px
- **Hover Effect**: Subtle color shift

### Inputs & Selectors
- **Background**: `rgba(255, 255, 255, 0.15)`
- **Border**: `1px solid rgba(255, 215, 0, 0.3)`
- **Focus State**: Gold border with glow
- **Border Radius**: 10px

### Progress Indicators
- **Bar Color**: `linear-gradient(90deg, #FFD700, #FFA500)`
- **Background**: Semi-transparent

## Animations & Effects

### Background Animation
- **Type**: Slow gradient shift
- **Duration**: 15 seconds
- **Effect**: Subtle color transition

### Hover Effects
- **Cards**: Scale up and elevate
- **Buttons**: Elevate and glow
- **Text**: Color transition

### Interactive Elements
- **Sliders**: Gold thumb with glow
- **Checkboxes**: Gold fill when checked
- **Expanders**: Smooth open/close

## Spacing & Layout

### Padding
- **Cards**: 25px
- **Sections**: 20px
- **Elements**: 15px
- **Text Blocks**: 12px

### Margins
- **Headers**: 25px bottom
- **Sections**: 30px bottom
- **Elements**: 10px
- **Cards**: 20px

### Grid System
- **Columns**: 3-column grid for metrics
- **Responsive Breakpoints**: Streamlit defaults
- **Gutters**: 20px

## Consistency Checklist

Before deploying any changes, ensure:
- [ ] All headers use the gold gradient text style
- [ ] Cards maintain consistent styling and shadows
- [ ] Buttons follow the gradient design pattern
- [ ] Inputs and selectors have uniform appearance
- [ ] Color palette is consistently applied
- [ ] Animations are smooth and not distracting
- [ ] Typography hierarchy is maintained
- [ ] Spacing follows the defined system

## Accessibility

- **Contrast**: Ensure text is readable against backgrounds
- **Focus States**: Clear focus indicators for keyboard navigation
- **Animations**: Subtle enough to not cause issues for users with vestibular disorders
- **Color Blindness**: Ensure information is not conveyed by color alone

## Future Enhancements

Consider these additions for future iterations:
- Dark/light mode toggle
- Custom theme selector
- Animation intensity controls
- Font size adjustment options