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

check_aslr() {
    local val
    val=$(cat /proc/sys/kernel/randomize_va_space 2>/dev/null)
    if [[ "$val" == "0" ]]; then
        echo "✅ ASLR is disabled."
    else
        echo "❌ ASLR is NOT disabled"
    fi
}

set_performance_governor() {
    cpupower frequency-set -g performance > /dev/null
    local core=$1
    local gov_file="/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor"
    local governor
    governor=$(cat "$gov_file")
    if [ "$governor" = "performance" ]; then
        echo "✅ core $core is set to performance mode."
        return 0
    else
        echo "❌ core $core is NOT set to performance mode (current: $governor)."
    fi
}

set_cpu_to_max_freq() {
    # Get the frequency limits line from cpupower output
    freq_info=$(cpupower -c "$1" frequency-info | grep "hardware limits")

    # Extract the upper frequency limit (handles decimal numbers correctly)
    max_freq=$(echo "$freq_info" | grep -oP ' - \K[0-9]+\.[0-9]+ [MG]Hz')

    # Set both upper and lower frequency limits to the max frequency
    sudo cpupower -c "$1" frequency-set -d "${max_freq// /}" -u "${max_freq// /}" > /dev/null

    echo "✅ core $1 set to fixed frequency: $max_freq"
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

    cpupower -c "$core" idle-set -D 0
    echo "✅ Deep idle states disabled"
}

check_isolated() {
    local core=$1
    local isolated
    isolated=$(cat /sys/devices/system/cpu/isolated)

    # Expand ranges like 8-15 into individual core numbers
    for token in ${isolated//,/ }; do
        if [[ $token == *-* ]]; then
            IFS=- read start end <<< "$token"
            for i in $(seq $start $end); do
                [[ $i -eq $core ]] && echo "✅ core $core is isolated" && return 0
            done
        else
            [[ $token -eq $core ]] && echo "✅ core $core is isolated" && return 0
        fi
    done

    echo "❌ core is NOT isolated"
}

# Check input
if [ -z "$1" ]; then
    echo "❌ No cores provided. Usage: $0 0,1,2,3"
    exit 1
fi

# Convert comma-separated list to space-separated and store in array
cores=(${1//,/ })

check_turbo
check_hwp
check_aslr
set_uncore_freq
prevent_swaps

for core in "${cores[@]}"; do
    set_performance_governor "$core"
    set_cpu_to_max_freq "$core"
    disable_deep_idle_states "$core"
    check_isolated "$core"
done

cset shield -e -- apptainer exec --writable-tmpfs --memory 120G --memory-swap 120G trireme.sif sh -c "cd trireme && taskset -c $1 $2"
