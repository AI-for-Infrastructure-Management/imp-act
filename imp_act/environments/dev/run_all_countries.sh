#!/bin/bash

# 1. Full list of countries (cleaned dataset - countries with more than 20 edges)
COUNTRIES=(
    "AT" "BE" "BG" "BY" "CH" "CZ" "DE" "DK" "EE" "ES"
    "FI" "FR" "GR" "HR" "HU" "IE" "IT" "LT" "LU" "LV"  
    "MK" "NL" "NO" "PL" "PT" "RO" "RS" "RU" "SE" "SI" 
    "SK" "TR" "UA" "UK"
)

# 2. Loop through the list
for country in "${COUNTRIES[@]}"; do
    echo "----------------------------------------"
    echo "Starting processing for: $country"
    echo "----------------------------------------"

    # 3. Run the commands in sequence
    # The '&&' ensures the next command only runs if the previous one succeeded.
    if python create_large_graph.py country="$country" && \
       python compute_reward_factor.py --preset "$country" && \
       python create_budget_scenarios.py preset="$country" dry_run=false overwrite=true; then
       
       echo "✅ Successfully completed all steps for $country"
    else
       echo "❌ Error encountered for $country. Skipping to next country..."
    fi

    echo "" # Empty line for readability
done

echo "🎉 Batch processing finished."