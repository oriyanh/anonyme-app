package com.mashehu.anonyme;

import androidx.fragment.app.FragmentActivity;
import androidx.navigation.Navigation;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;

public class ShareActivity extends FragmentActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_share);

        assert getIntent() != null;
        Bundle args = new Bundle();
        ArrayList<Uri> arguments = getIntent().getParcelableArrayListExtra(Intent.EXTRA_STREAM);
        ArrayList<String> files = new ArrayList<>();
        if ((Intent.ACTION_SEND.equals(getIntent().getAction())) ||
                (Intent.ACTION_SEND_MULTIPLE.equals(getIntent().getAction())))
        {
            assert arguments != null;
            for (Uri argument: arguments)
            {
                files.add(argument.getPath());
            }
            args.putStringArrayList(IMAGE_DIRS_ARGUMENT_KEY, files);
            Navigation.findNavController(findViewById(R.id.confirm_images_layout)).navigate(
                    R.id.confirmImagesFragment, args);
        }
    }
}
