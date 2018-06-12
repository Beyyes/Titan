#!/bin/bash

for c in `printenv | perl -sne 'print "$1 " if m/^SPARK_CONF_(.+?)=.*/'`; do
    name=`echo ${c} | perl -pe 's/___/-/g; s/__/_/g; s/_/./g'`
    var="SPARK_CONF_${c}"
    value=${!var}
    echo "Setting SPARK property $name=$value"
    echo $name $value >> $SPARK_HOME/conf/spark-defaults.conf
done

for c in `printenv | perl -sne 'print "$1 " if m/^GRAPHITE_CONF_(.+?)=.*/'`; do
    var="GRAPHITE_CONF_${c}"
    value=${!var}
    echo "Setting GRAPHITE property $name=$value"
    sed -i "s/$var/$value/g" /metrics.properties
done
if [ -z "`grep 'GRAPHITE_CONF_' /metrics.properties`" ]; then mv /metrics.properties $SPARK_HOME/conf/metrics.properties; fi

exec /entrypoint.sh $@
