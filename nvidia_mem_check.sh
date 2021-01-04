while [ true ]; do
  DPID=`ps -ef | grep -v grep | grep "python main.py"`
  if [ "$DPID" != "" ]; then
      nvidia-smi
  else
      echo " down"
      exit 0
  fi
  sleep 1
done

