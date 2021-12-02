import React, { useState, useEffect } from 'react';
import { useLocation, useHistory } from 'react-router-dom';

import { CircularProgress } from '@material-ui/core';

import type {
  ContributorRating,
  PaginatedContributorRatingList,
} from 'src/services/openapi';
import Pagination from 'src/components/Pagination';
import VideoList from 'src/features/videos/VideoList';
import { UsersService } from 'src/services/openapi';
import { ContentBox } from 'src/components';
import { getPublicStatusAction } from 'src/features/videos/PublicStatusAction';

function VideoRatingsPage() {
  const prov: PaginatedContributorRatingList = {
    count: 0,
    results: [],
  };
  const [ratings, setRatings] = useState(prov);
  const [isLoading, setIsLoading] = useState(true);
  const location = useLocation();
  const history = useHistory();
  const searchParams = new URLSearchParams(location.search);
  const limit = 20;
  const offset = Number(searchParams.get('offset') || 0);
  const videoCount = ratings.count || 0;

  function handleOffsetChange(newOffset: number) {
    searchParams.delete('offset');
    searchParams.append('offset', newOffset.toString());
    history.push('/recommendations/?' + searchParams.toString());
  }

  useEffect(() => {
    setIsLoading(true);
    const loadData = async () => {
      const response = await UsersService.usersMeContributorRatingsList(
        /* isPublic */ undefined,
        limit,
        offset
      );
      setRatings(response);
      setIsLoading(false);
    };
    loadData();
  }, [offset]);

  const videos = (ratings.results || []).map(
    (rating: ContributorRating) => rating.video
  );

  const idToRating = Object.fromEntries(
    (ratings.results || []).map((rating) => [rating.video.video_id, rating])
  );

  const getPublicStatus = (videoId: string) => idToRating[videoId];

  return (
    <ContentBox noMinPadding maxWidth="md">
      {isLoading ? (
        <CircularProgress />
      ) : (
        <VideoList
          videos={videos}
          settings={[getPublicStatusAction(getPublicStatus)]}
        />
      )}
      {!isLoading && videoCount > 0 && (
        <Pagination
          offset={offset}
          count={videoCount}
          onOffsetChange={handleOffsetChange}
          limit={limit}
        />
      )}
    </ContentBox>
  );
}

export default VideoRatingsPage;
