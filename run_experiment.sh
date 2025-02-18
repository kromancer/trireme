#!/bin/bash

check_turbo() {
    if [ -f "/sys/devices/system/cpu/intel_pstate/no_turbo" ]; then
        turbo_status=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        if [ "$turbo_status" -eq 1 ]; then
            echo "✅ Turbo Boost is DISABLED (intel_pstate)."
        else
            echo "❌ Turbo Boost is ENABLED (intel_pstate)."
            exit 1
        fi
    fi
}

set_performance_governor() {
    cpupower frequency-set -g performance > /dev/null
    local core=$1
    local gov_file="/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor"
    local governor=$(cat "$gov_file")
    if [ "$governor" = "performance" ]; then
        echo "✅ CPU core $core is set to performance mode."
        return 0
    else
        echo "❌ CPU core $core is NOT set to performance mode (current: $governor)."
        return 1
    fi
}

set_cpu_to_max_freq() {
    local core=$1

    # Get the frequency limits line from cpupower output
    local freq_info=$(cpupower -c "$core" frequency-info | grep "hardware limits")

    # Extract the upper frequency limit (handles decimal numbers correctly)
    local max_freq=$(echo "$freq_info" | grep -oP ' - \K[0-9]+\.[0-9]+ [MG]Hz')

    # Set both upper and lower frequency limits to the max frequency
    sudo cpupower -c "$core" frequency-set -d "${max_freq// /}" -u "${max_freq// /}" > /dev/null

    echo "✅ CPU core $core set to fixed frequency: ${max_freq}"
}

prevent_swaps() {
    sysctl -w vm.overcommit_memory=0
    echo "✅ vm.overcommit_memory=0"

    sysctl vm.swappiness=0
    echo "✅ sysctl vm.swappiness=0"
}

check_turbo
set_performance_governor $1
set_cpu_to_max_freq $1
prevent_swaps

cset shield -e -- apptainer exec trireme.sif sh -c "cd trireme && taskset -c $1 ./experiment.sh"
