package com.mashehu.anonyme;

import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;

import com.mashehu.anonyme.fragments.AppViewModel;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Utilities.isProcessing;

public class ShareActivity extends FragmentActivity {

    private static final String TAG = "anonyme.ShareActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_share);

        assert getIntent() != null;
        ArrayList<Uri> arguments = new ArrayList<>();
        ArrayList<String> files = new ArrayList<>();
        if (Intent.ACTION_SEND_MULTIPLE.equals(getIntent().getAction()))
        {
            arguments = getIntent().getParcelableArrayListExtra(Intent.EXTRA_STREAM);
        }
        else if (Intent.ACTION_SEND.equals(getIntent().getAction()))
        {
            Uri arg = getIntent().getParcelableExtra(Intent.EXTRA_STREAM);
            arguments.add(arg);
        }
        else
        {
            finish();
        }

        assert arguments != null;
        for (Uri argument: arguments)
        {
            assert argument.getPath() != null;
            String filePath;
//            String wholeID = DocumentsContract.getDocumentId(argument);

            // Split at colon, use second item in the array
//            String id = wholeID.split(":")[1];

//            String[] column = { MediaStore.Images.Media.DATA };

            // where id is equal to
//            String sel = MediaStore.Images.Media._ID + "=?";

            Cursor cursor = this.getContentResolver().query(argument, null, null, null, null);
            if (cursor == null)
            {
                filePath = argument.getPath();
            }
            else
            {
                cursor.moveToFirst();
                int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
                filePath = cursor.getString(idx);
            }
            files.add(filePath);
        }

        AppViewModel viewModel = ViewModelProviders.of(this).get(AppViewModel.class);
        for (String img : files) {
            viewModel.addImage(img);
        }
        viewModel.setCurrentTab(-1);
    }
}
