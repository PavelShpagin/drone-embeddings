#!/bin/bash

# Push SuperPoint Weights to GitHub
echo "🚀 Pushing SuperPoint Weights (20 epochs)"
echo "📊 Current SuperPoint weights:"
echo ""

# Show current weights
ls -lh superpoint_uav_trained/
echo ""

# Add SuperPoint weights to git
echo "📥 Adding SuperPoint weights to git..."
git add superpoint_uav_trained/*.pth
git add .gitignore

# Commit
echo "💾 Committing SuperPoint weights..."
git commit -m "Add SuperPoint weights (20 epochs) for mobile Cursor access

- SuperPoint trained for 20 epochs on UAV data
- Model size: ~5MB each (Pi Zero friendly)
- Added exception in .gitignore for SuperPoint weights
- Enables access from mobile Cursor IDE"

# Push
echo "🚀 Pushing to GitHub..."
git push

echo ""
echo "✅ SuperPoint weights pushed successfully!"
echo "📱 Now accessible from mobile Cursor IDE"
echo ""
echo "📊 Available weights:"
echo "   - superpoint_uav_epoch_5.pth (5MB)"
echo "   - superpoint_uav_epoch_10.pth (5MB)"
echo "   - superpoint_uav_epoch_15.pth (5MB)"
echo "   - superpoint_uav_epoch_20.pth (5MB)"
echo "   - superpoint_uav_final.pth (5MB)" 