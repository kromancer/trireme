#!/bin/bash

check_turbo() {
    if [ -f "/sys/devices/system/cpu/intel_pstate/no_turbo" ]; then
        turbo_status=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        if [ "$turbo_status" -eq 1 ]; then
            echo "✅ Turbo Boost is DISABLED (intel_pstate)."
        else
            echo "❌ Turbo Boost is ENABLED (intel_pstate)."
        fi
    fi
}

check_hwp() {
    if [ -f "/sys/devices/system/cpu/intel_pstate/status" ]; then
        hwp_status=$(cat /sys/devices/system/cpu/intel_pstate/status)
        if [ "$hwp_status" = "active" ]; then
            echo "✅ HWP is disabled."
        else
            echo "❌ HWP is NOT disabled: $hwp_status"
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
    fi
}

set_cpu_to_max_freq() {
    # Get the frequency limits line from cpupower output
    freq_info=$(cpupower -c "$1" frequency-info | grep "hardware limits")

    # Extract the upper frequency limit (handles decimal numbers correctly)
    max_freq=$(echo "$freq_info" | grep -oP ' - \K[0-9]+\.[0-9]+ [MG]Hz')

    # Set both upper and lower frequency limits to the max frequency
    sudo cpupower -c "$1" frequency-set -d "${max_freq// /}" -u "${max_freq// /}" > /dev/null

    echo "✅ CPU core $1 set to fixed frequency: $max_freq"
}

set_uncore_freq() {
    base_path="/sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00"
    min_freq_file="$base_path/min_freq_khz"
    max_freq_file="$base_path/max_freq_khz"
    init_max_freq_file="$base_path/initial_max_freq_khz"

    # Check if the initial_max_freq_khz file exists
    if [[ -f "$init_max_freq_file" ]]; then
        # Read the frequency value from initial_max_freq_khz
        init_freq=$(cat "$init_max_freq_file")

        # Write the frequency value to min and max freq files
        echo "$init_freq" > "$min_freq_file"
        echo "$init_freq" > "$max_freq_file"

        echo "✅ Uncore frequency set to ${init_freq} kHz."
    else
	echo "❌ initial_max_freq_khz file not found!"
    fi
}

prevent_swaps() {
    sysctl -w vm.overcommit_memory=0 > /dev/null
    echo "✅ vm.overcommit_memory=0"

    sysctl vm.swappiness=0 > /dev/null
    echo "✅ sysctl vm.swappiness=0"
}

disable_deep_idle_states() {
    local core=$1

    cpupower -c $core idle-set -D 0
    echo "✅ Deep idle states disabled"
}

check_turbo
check_hwp
set_performance_governor $1
set_cpu_to_max_freq $1
set_uncore_freq
prevent_swaps
disable_deep_idle_states $1

cset shield -e -- apptainer exec --writable-tmpfs --memory 120G --memory-swap 120G trireme.sif sh -c "cd trireme && taskset -c $1 ./experiment.sh"
