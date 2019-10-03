package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentStatePagerAdapter;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager.widget.ViewPager;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.CameraCaptureContainerFragment;
import com.mashehu.anonyme.fragments.GalleryContainerFragment;

import java.util.ArrayList;
import java.util.List;

import static android.view.View.GONE;
import static android.view.View.INVISIBLE;
import static android.view.View.VISIBLE;

public class RecyclerUtils {

	// adapters
	public static class ConfirmImagesAdapter extends ListAdapter<ImageData, PreviewImageHolder> {
		private ArrayList<ImageData> images;
		private Context context;
		public PreviewItemsCallback callback;

		public ConfirmImagesAdapter(Context context, ArrayList<ImageData> images) {
			super(new PreviewDiffCallback());
			this.images = images;
			this.context = context;
		}


		@NonNull
		@Override
		public PreviewImageHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
			View view = LayoutInflater.from(parent.getContext())
					.inflate(R.layout.item_image_preview, parent, false);
			return new PreviewImageHolder(view);
		}

		@Override
		public void onBindViewHolder(@NonNull PreviewImageHolder holder, int position) {
			//		holder.imageView.setOnClickListener(v -> {
			//			if (!multipleSelectionMode) {
			//				Bundle args = new Bundle();
			//				String imageView = images.get(position).getImagePath();
			//				ArrayList<String> images = new ArrayList<>();
			//				images.add(imageView);
			//				args.putStringArrayList("imageDirs", images);
			//				Navigation.findNavController(v).navigateIfNecessary(
			//						R.id.action_galleryFragment_to_confirmImagesFragment,
			//						args);
			//			}
			//			else {} //TODO work on multiple selection mode
			//		});
			//		holder.imageView.setOnLongClickListener(v -> true);

			GlideApp.with(context)
					.load(images.get(position).getImagePath())
					.confirmImagesLarge()
					.into(holder.img);
		}

		public Context getContext() {
			return context;
		}

		public void deleteItem(int position) {
			if (callback != null) {
				callback.removeItem(images.get(position));
				notifyItemRemoved(position);
			}
		}

		@Override
		public void submitList(@Nullable List<ImageData> list) {
			images = (ArrayList<ImageData>) list;
			super.submitList(list);
		}


		@Override
		public int getItemCount() {
			return images.size();
		}


	}

	public static class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailViewHolder> {
		private static final String TAG = "anonyme.ThumbnailAdapter";
		private ArrayList<ImageData> imageList;
		private Context context;
		private ThumbnailCallback callback;

		public ThumbnailAdapter(Context context, ThumbnailCallback callback, ArrayList<ImageData> images) {
			this.imageList = images;
			this.context = context;
			this.callback = callback;
		}

		@NonNull
		@Override
		public ThumbnailViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
			View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_image_thumbnail, parent, false);
			return new RecyclerUtils.ThumbnailViewHolder(view);
		}

		@Override
		public void onBindViewHolder(@NonNull ThumbnailViewHolder holder, int position) {
			ImageData img = imageList.get(position);

			holder.imageView.setOnClickListener(v -> {
				Log.d(TAG, "Normal click on image");
				if (callback.isMultipleSelection()) {
					holder.toggleCheckbox();
					if (holder.checkbox.getVisibility() == INVISIBLE) {
						callback.removeImage(img); // means after toggling, image is no longer selected
					}

					else {
						callback.addImage(img);
					}
				}

				else {
					callback.addImage(img);
					callback.startProcessing(v);
				}
			});

			holder.imageView.setOnLongClickListener(v -> {
				Log.d(TAG, "Long click on image");

				callback.setMultipleSelection(true);
				holder.toggleCheckbox();
				if (holder.checkbox.getVisibility() == INVISIBLE) {
					callback.removeImage(img); // means after toggling, image is no longer selected
				}

				else {
					callback.addImage(img);
				}

				return true;
			});

			GlideApp.with(context)
					.load(img.getImagePath())
					.galleryThumbnail()
					.into(holder.imageView);
		}

		@Override
		public int getItemCount() {
			return imageList.size();
		}

	}

	public static class FragmentPagerAdapter extends FragmentStatePagerAdapter {

		public FragmentPagerAdapter(@NonNull FragmentManager fm, int behavior) {
			super(fm, behavior);

		}

		@NonNull
		@Override
		public Fragment getItem(int position) {
			if (position == 1) {
				return new CameraCaptureContainerFragment();
			}
			return new GalleryContainerFragment();
		}

		@Override
		public int getCount() {
			return 2;
		}
	}


	// data holders
	public static class ImageData {
		private String imagePath;

		public String getImagePath() {
			return imagePath;
		}

		public void setImagePath(String imagePath) {
			this.imagePath = imagePath;
		}
	}

	public static class ThumbnailViewHolder extends RecyclerView.ViewHolder {
		public ImageView imageView;
		public ImageView checkbox;

		public ThumbnailViewHolder(@NonNull View itemView) {
			super(itemView);
			imageView = itemView.findViewById(R.id.thumbnailView);
			checkbox = itemView.findViewById(R.id.thumbnailCheckbox);
			checkbox.setVisibility(INVISIBLE);
		}

		public void toggleCheckbox() {
			switch (checkbox.getVisibility()) {
				case VISIBLE:
					checkbox.setVisibility(INVISIBLE);
					break;
				case INVISIBLE:
					checkbox.setVisibility(VISIBLE);
				case GONE:
				default: // do nothing
			}
		}
	}

	public static class PreviewImageHolder extends RecyclerView.ViewHolder {
		public ImageView img;

		public PreviewImageHolder(@NonNull View itemView) {
			super(itemView);
			img = itemView.findViewById(R.id.confirm_image_large_view);
		}
	}

	// interfaces and callbacks
	public static class SwipeToDeleteCallback extends ItemTouchHelper.SimpleCallback {
		private ConfirmImagesAdapter adapter;

		public SwipeToDeleteCallback(ConfirmImagesAdapter adapter) {
			super(0, ItemTouchHelper.UP);
			this.adapter = adapter;
			//		icon = ContextCompat.getDrawable(this.adapter.getContext(),
			//				R.drawable.ic_delete_white_36);
			//		background = new ColorDrawable(Color.RED);
		}

		@Override
		public void onChildDraw(@NonNull Canvas c, @NonNull RecyclerView recyclerView,
								@NonNull RecyclerView.ViewHolder viewHolder,
								float dX, float dY, int actionState, boolean isCurrentlyActive) {
			final View foregroundView = ((PreviewImageHolder) viewHolder).img;
			getDefaultUIUtil().onDraw(c, recyclerView, foregroundView, dX, dY,
					actionState, isCurrentlyActive);

			//		View itemView = viewHolder.itemView;
			//		int backgroundCornerOffset = 20;
			//		int iconMargin = (itemView.getHeight() - icon.getIntrinsicHeight()) / 2;
			//		int iconTop = itemView.getTop() + (itemView.getHeight() - icon.getIntrinsicHeight()) / 2;
			//		int iconBottom = iconTop + icon.getIntrinsicHeight();
			//
			//		if (dY > 0) { // Swiping upwards
			//			int iconLeft = itemView.getLeft() + iconMargin + icon.getIntrinsicWidth();
			//			int iconRight = itemView.getLeft() + iconMargin;
			//			icon.setBounds(iconLeft, iconTop, iconRight, iconBottom);
			//			background.setBounds(itemView.getLeft(), itemView.getTop() + (int) dY,
			//					itemView.getRight(),
			//					itemView.getBottom() + backgroundCornerOffset);
			//
			//		}
			//		else { // view is unSwiped
			//			background.setBounds(0, 0, 0, 0);
			//		}
			//		background.draw(c);
			//		icon.draw(c);
		}

		@Override
		public boolean onMove(@NonNull RecyclerView recyclerView, @NonNull RecyclerView.ViewHolder viewHolder, @NonNull RecyclerView.ViewHolder target) {
			return true;
		}

		@Override
		public void onSelectedChanged(@Nullable RecyclerView.ViewHolder viewHolder, int actionState) {
			if (viewHolder != null) {
				final View foregroundView = ((PreviewImageHolder) viewHolder).img;

				getDefaultUIUtil().onSelected(foregroundView);
			}
		}

		@Override
		public void onChildDrawOver(@NonNull Canvas c, @NonNull RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder, float dX, float dY, int actionState, boolean isCurrentlyActive) {
			final View foregroundView = ((PreviewImageHolder) viewHolder).img;
			getDefaultUIUtil().onDrawOver(c, recyclerView, foregroundView, dX, dY,
					actionState, isCurrentlyActive);
		}

		@Override
		public void clearView(@NonNull RecyclerView recyclerView, @NonNull RecyclerView.ViewHolder viewHolder) {
			final View foregroundView = ((PreviewImageHolder) viewHolder).img;
			getDefaultUIUtil().clearView(foregroundView);
		}

		@Override
		public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
			int position = viewHolder.getAdapterPosition();
			Log.d("anonyme.SwipeToDeleteCallback", "Swipe");
			adapter.deleteItem(position);
		}
	}

	public static class PreviewDiffCallback extends DiffUtil.ItemCallback<ImageData> {
		@Override
		public boolean areItemsTheSame(@NonNull ImageData oldItem, @NonNull ImageData newItem) {
			return oldItem == newItem;
		}

		@Override
		public boolean areContentsTheSame(@NonNull ImageData oldItem, @NonNull ImageData newItem) {
			return oldItem.getImagePath().equals(newItem.getImagePath());
		}
	}

	public interface PreviewItemsCallback {

		public void removeItem(ImageData img);
	}

	public interface ThumbnailCallback {
		public void addImage(ImageData imageData);

		public void removeImage(ImageData imageData);

		public void startProcessing(View v);

		public boolean isMultipleSelection();

		public void setMultipleSelection(boolean isMultipleSelection);
	}
}
