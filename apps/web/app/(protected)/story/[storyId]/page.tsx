import NewsPreview from '@/components/specific/news-scroll/preview/news-preview';
import PreviewHeader from '@/components/specific/news-scroll/preview/preview-header';

const stories = [
	{
		image:
			'https://ko77xaoqa4.ufs.sh/f/tCV5HvjhrFj7IdoJ5zuQoRFHJr4sv5UPGOuXKthkE83bW0i1',
		title:
			'SpaceX letter criticizes FAA for “systemic challenges” in launch licensing',
		article:
			"SpaceX is challenging a $633,000 FAA fine for two 2023 launch violations, arguing the changes were minor and didn’t affect safety. The company claims the FAA was slow to approve updates and suggests the fines are politically motivated. CEO Elon Musk vowed to sue but hasn't yet.",
		tags: ['Space', 'Elon Musk', 'FAA'],
		publishedAt: '12 hours ago',
		views: '43k',
		slug: 'spacex-letter-criticizes-faa-for-systemic-challenges-in-launch-licensing-124H8',
		source: {
			name: 'CNN',
			image:
				'https://ko77xaoqa4.ufs.sh/f/tCV5HvjhrFj7Al0k2KoQ2M0UGsxT3Li1eYlgDhHwnmPkprWI',
			url: 'https://www.cnn.com/2023/09/05/tech/spacex-faa-fine-elon-musk-lawsuit/index.html',
		},
	},
	{
		image: 'https://utfs.io/f/tCV5HvjhrFj7e9NgBZxqDOE9pI4s2uQrfZnahoL6R8vM03SV',
		title: 'Lockheed Martin challenges narrative on GPS vulnerability',
		publishedAt: '12 hours ago',
		views: '43k',
		slug: 'lockheed-martin-challenges-narrative-on-gps-vulnerability-124H8',
		article:
			"SpaceX is challenging a $633,000 FAA fine for two 2023 launch violations, arguing the changes were minor and didn’t affect safety. The company claims the FAA was slow to approve updates and suggests the fines are politically motivated. CEO Elon Musk vowed to sue but hasn't yet",
		tags: ['Space', 'Technology', 'Science'],
		source: {
			name: 'CNN',
			image:
				'https://ko77xaoqa4.ufs.sh/f/tCV5HvjhrFj7Al0k2KoQ2M0UGsxT3Li1eYlgDhHwnmPkprWI',
			url: 'https://www.cnn.com/2023/09/05/tech/spacex-faa-fine-elon-musk-lawsuit/index.html',
		},
	},
	{
		image: 'https://utfs.io/f/tCV5HvjhrFj78ppsNbrmDKZ6LVIqCe2WQuHiMNX41BYwsvEd',
		title: 'India eyes record year for space with 10 planned launches',
		publishedAt: '5 hours ago',
		views: '28k',
		slug: 'india-eyes-record-year-for-space-with-10-planned-launches-GJ878G',
		article:
			"SpaceX is challenging a $633,000 FAA fine for two 2023 launch violations, arguing the changes were minor and didn’t affect safety. The company claims the FAA was slow to approve updates and suggests the fines are politically motivated. CEO Elon Musk vowed to sue but hasn't yet",
		tags: ['Space', 'Technology', 'Science'],
		source: {
			name: 'CNN',
			image:
				'https://ko77xaoqa4.ufs.sh/f/tCV5HvjhrFj7Al0k2KoQ2M0UGsxT3Li1eYlgDhHwnmPkprWI',
			url: 'https://www.cnn.com/2023/09/05/tech/spacex-faa-fine-elon-musk-lawsuit/index.html',
		},
	},
];

const StoryPage = () => {
	return (
		<main className="overflow-y-auto overflow-x-hidden snap-y snap-mandatory h-screen w-screen transform-gpu">
			<PreviewHeader />
			{stories.map((story, index) => (
				<NewsPreview
					key={story.slug}
					image={story.image}
					title={story.title}
					article={story.article}
					tags={story.tags}
					publishedAt={story.publishedAt}
					source={story.source}
					reads={story.views}
					index={index}
				/>
			))}
		</main>
	);
};

export default StoryPage;
