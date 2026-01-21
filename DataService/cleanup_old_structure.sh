#!/bin/bash
# Migration script to clean up old DataService files after restructure
# Run this after verifying the new structure works correctly

echo "DataService Restructure - Cleanup Script"
echo "========================================="
echo ""
echo "This will remove the old top-level files that have been moved to submodules:"
echo "  - metro_models.py → core/models.py"
echo "  - metro_data_generator.py → generators/metro_generator.py"
echo "  - schedule_optimizer.py → optimizers/schedule_optimizer.py"
echo "  - api.py → api/routes.py"
echo "  - enhanced_generator.py → generators/enhanced_generator.py"
echo "  - synthetic_base.py → generators/synthetic_base.py"
echo "  - synthetic_extend.py → generators/synthetic_extend.py"
echo ""
echo "The new files are already in place and tested."
echo ""

read -p "Do you want to proceed with cleanup? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

cd "$(dirname "$0")"

# Backup old files first
echo ""
echo "Creating backup directory..."
mkdir -p .old_structure_backup
cp metro_models.py .old_structure_backup/ 2>/dev/null
cp metro_data_generator.py .old_structure_backup/ 2>/dev/null
cp schedule_optimizer.py .old_structure_backup/ 2>/dev/null
cp api.py .old_structure_backup/ 2>/dev/null
cp enhanced_generator.py .old_structure_backup/ 2>/dev/null
cp synthetic_base.py .old_structure_backup/ 2>/dev/null
cp synthetic_extend.py .old_structure_backup/ 2>/dev/null

echo "✓ Backup created in .old_structure_backup/"

# Remove old files
echo ""
echo "Removing old files..."
rm -f metro_models.py
rm -f metro_data_generator.py
rm -f schedule_optimizer.py
rm -f api.py
rm -f enhanced_generator.py
rm -f synthetic_base.py
rm -f synthetic_extend.py

echo "✓ Old files removed"
echo ""
echo "Cleanup complete!"
echo ""
echo "New structure:"
echo "  DataService/"
echo "  ├── core/          (models, utils)"
echo "  ├── generators/    (data generation)"
echo "  ├── optimizers/    (schedule optimization)"
echo "  └── api/           (REST endpoints)"
echo ""
echo "Backups are in: DataService/.old_structure_backup/"
