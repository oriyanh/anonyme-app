package com.mashehu.anonyme.services;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;

public class EngineService extends Service {
    public EngineService() {
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        startForeground(1, new Notification());
        return super.onStartCommand(intent, flags, startId);
    }
}
