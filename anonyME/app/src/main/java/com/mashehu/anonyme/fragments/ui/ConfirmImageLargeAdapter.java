package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.EmptyListCallback;

import java.util.ArrayList;

public class ConfirmImageLargeAdapter extends RecyclerView.Adapter<ConfirmImageLargeAdapter.ConfirmImageViewHolder> {
	private ArrayList<ImageData> images;
	private Context context;
	private EmptyListCallback callback;

	public ConfirmImageLargeAdapter(Context context, ArrayList<ImageData> images, EmptyListCallback callback) {
		this.images = images;
		this.context = context;
		this.callback = callback;
	}


	@NonNull
	@Override
	public ConfirmImageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
		View view = LayoutInflater.from(parent.getContext())
				.inflate(R.layout.item_confirm_image_large, parent, false);
		return new ConfirmImageViewHolder(view);
	}

	@Override
	public void onBindViewHolder(@NonNull ConfirmImageViewHolder holder, int position) {
//		holder.img.setOnClickListener(v -> {
//			if (!multipleSelectionMode) {
//				Bundle args = new Bundle();
//				String img = images.get(position).getImagePath();
//				ArrayList<String> images = new ArrayList<>();
//				images.add(img);
//				args.putStringArrayList("imageDirs", images);
//				Navigation.findNavController(v).navigate(
//						R.id.action_galleryFragment_to_confirmImagesFragment,
//						args);
//			}
//			else {} //TODO work on multiple selection mode
//		});
//		holder.img.setOnLongClickListener(v -> true);

		GlideApp.with(context)
				.load(images.get(position).getImagePath())
				.confirmImagesLarge()
				.into(holder.img);
	}

	@NonNull
	public ArrayList<String> getImagePaths() {
		ArrayList<String> paths = new ArrayList<>(images.size());
		for (ImageData img : images) {
			paths.add(img.getImagePath());
		}
		return paths;
	}

	public Context getContext() {
		return context;
	}

	public void deleteItem(int position) {
		images.remove(position);
		notifyItemRemoved(position);
		callback.handleEmptyList();  // todo replace with LiveData observers, better practice
	}

	@Override
	public int getItemCount() {
		return images.size();
	}

	public static class ConfirmImageViewHolder extends RecyclerView.ViewHolder {
		public ImageView img;

		public ConfirmImageViewHolder(@NonNull View itemView) {
			super(itemView);
			img = itemView.findViewById(R.id.confirm_image_large_view);
		}
	}
}
